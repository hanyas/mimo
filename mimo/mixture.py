import copy

import numpy as np
import scipy.special as special
from scipy.special import logsumexp

from mimo.distribution import Distribution, Conditional

from mimo.distributions.bayesian import BayesianCategoricalWithDirichlet
from mimo.distributions.bayesian import BayesianCategoricalWithStickBreaking

from mimo.util.text import progprint_xrange
from mimo.util.general import sample_discrete_from_log

from mimo.util.general import multivariate_studentt_loglik as mvt_logpdf
from mimo.util.general import multivariate_gaussian_loglik as mvn_logpdf

import pathos
from pathos.pools import _ProcessPool as Pool
nb_cores = pathos.multiprocessing.cpu_count()


def pass_obs_arg(f):
    def wrapper(self, obs=None, **kwargs):
        if obs is None:
            assert self.has_data()
            obs = [_obs for _obs in self.obs]
        else:
            obs = obs if isinstance(obs, list) else [obs]

        return f(self, obs, **kwargs)
    return wrapper


def pass_obs_and_labels(f):
    def wrapper(self, obs=None, labels=None, **kwargs):
        if obs is None or labels is None:
            assert self.has_data()
            obs = [_obs for _obs in self.obs]
            labels = self.labels
        else:
            obs = obs if isinstance(obs, list) else [obs]
            labels = [self.gating.rvs(len(_obs)) for _obs in obs]\
                if labels is None else labels

        return f(self, obs, labels, **kwargs)
    return wrapper


def pass_target_and_input_arg(f):
    def wrapper(self, y=None, x=None, **kwargs):
        if y is None or x is None:
            assert self.has_data()
            y = [_y for _y in self.target]
            x = [_x for _x in self.input]
        else:
            y = y if isinstance(y, list) else [y]
            x = x if isinstance(x, list) else [x]

        return f(self, y, x, **kwargs)
    return wrapper


def pass_target_input_and_labels_arg(f):
    def wrapper(self, y=None, x=None, z=None, **kwargs):
        if y is None or x is None and z is None:
            assert self.has_data()
            y = [_y for _y in self.target]
            x = [_x for _x in self.input]
            z = self.labels
        else:
            y = y if isinstance(y, list) else [y]
            x = x if isinstance(x, list) else [x]
            z = [self.gating.rvs(len(_y)) for _y in y]\
                if z is None else z

        return f(self, y, x, z, **kwargs)
    return wrapper


class BayesianMixtureOfGaussians(Distribution):
    """
    This class is for mixtures of other distributions.
    """

    def __init__(self, gating, components):
        assert len(components) > 0

        self.gating = gating
        self.components = components

        self.obs = []
        self.labels = []

        self.whitend = False
        self.obs_transform = None

    @property
    def nb_params(self):
        return sum(c.nb_params for c in self.components) + self.gating.nb_params

    @property
    def size(self):
        return len(self.components)

    @property
    def used_labels(self):
        assert self.has_data()
        label_usages = sum(np.bincount(_label, minlength=self.size) for _label in self.labels)
        used_labels, = np.where(label_usages > 0)
        return used_labels

    def add_data(self, obs, whiten=False):
        obs = obs if isinstance(obs, list) else [obs]
        for _obs in obs:
            self.labels.append(self.gating.rvs(len(_obs)))

        if whiten:
            self.whitend = True
            from sklearn.decomposition import PCA

            data = np.vstack([_obs for _obs in obs])
            self.obs_transform = PCA(n_components=data.shape[-1], whiten=True)
            self.obs_transform.fit(data)
            for _obs in obs:
                self.obs.append(self.obs_transform.transform(_obs))
        else:
            self.obs = obs

    def clear_data(self):
        self.obs.clear()
        self.labels.clear()

    def clear_transform(self):
        self.whitend = False
        self.obs_transform = None

    def has_data(self):
        return len(self.obs) > 0

    def rvs(self, size=1):
        labels = self.gating.rvs(size)
        counts = np.bincount(labels, minlength=self.size)

        obs = np.empty((size, self.dim))
        for idx, (c, count) in enumerate(zip(self.components, counts)):
            obs[labels == idx, ...] = c.rvs(count)

        perm = np.random.permutation(size)
        obs, z = obs[perm], labels[perm]

        return obs, labels

    def log_likelihood(self, obs):
        assert isinstance(obs, (np.ndarray, list))
        if isinstance(obs, list):
            return sum(self.log_likelihood(_obs) for _obs in obs)
        else:
            scores = self.log_scores(obs)
            return np.sum(logsumexp(scores[~np.isnan(obs).any(1)], axis=1))

    def mean(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def log_partition(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def log_scores(self, obs):
        N, K = obs.shape[0], self.size

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, c in enumerate(self.components):
            component_scores[:, idx] = c.log_likelihood(obs)
        component_scores = np.nan_to_num(component_scores)

        gating_scores = self.gating.log_likelihood(np.arange(K))
        score = gating_scores + component_scores
        return score

    # Expectation-Maximization
    def scores(self, obs):
        logr = self.log_scores(obs)
        score = np.exp(logr - np.max(logr, axis=1, keepdims=True))
        score /= np.sum(score, axis=1, keepdims=True)
        return score

    @pass_obs_arg
    def max_likelihood(self, obs=None):
        # Expectation step
        scores = []
        for _obs in obs:
            scores.append(self.scores(_obs))

        # Maximization step
        for idx, c in enumerate(self.components):
            c.max_likelihood([_obs for _obs in obs],
                             [_score[:, idx] for _score in scores])

        # mixture weights
        self.gating.max_likelihood(None, [_score for _score in scores])

    # Gibbs sampling
    @pass_obs_and_labels
    def resample(self, obs=None, labels=None):
        self._resample_components(obs, labels)
        self._resample_gating(labels)
        labels = self._resample_labels(obs)

        if self.has_data():
            self.labels = labels

    def copy_sample(self):
        new = copy.copy(self)
        new.components = [c.copy_sample() for c in self.components]
        new.gating = self.gating.copy_sample()
        return new

    def _resample_components(self, obs, labels):
        for idx, c in enumerate(self.components):
            c.resample(data=[_obs[_label == idx]
                             for _obs, _label in zip(obs, labels)])

    def _resample_gating(self, labels):
        self.gating.resample([_label for _label in labels])

    def _resample_labels(self, obs):
        labels = []
        for _obs in obs:
            score = self.log_scores(_obs)
            labels.append(sample_discrete_from_log(score, axis=1))
        return labels

    # Mean Field
    def expected_scores(self, obs):
        N, K = obs.shape[0], self.size

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, c in enumerate(self.components):
            component_scores[:, idx] = c.expected_log_likelihood(obs)
        component_scores = np.nan_to_num(component_scores)

        if isinstance(self.gating, BayesianCategoricalWithDirichlet):
            gating_scores = self.gating.expected_log_likelihood(np.arange(K))
        elif isinstance(self.gating, BayesianCategoricalWithStickBreaking):
            E_log_stick, E_log_rest = self.gating.expected_log_likelihood(np.arange(K))
            gating_scores = np.take(E_log_stick + np.hstack((0, np.cumsum(E_log_rest)[:-1])), np.arange(K))
        else:
            raise NotImplementedError

        logr = gating_scores + component_scores

        r = np.exp(logr - np.max(logr, axis=1, keepdims=True))
        r /= np.sum(r, axis=1, keepdims=True)

        return r

    def meanfield_coordinate_descent(self, tol=1e-1, maxiter=250, progprint=False):
        elbo = []
        step_iterator = range(maxiter) if not progprint else progprint_xrange(maxiter)
        for _ in step_iterator:
            elbo.append(self.meanfield_update())
            if elbo[-1] is not None and len(elbo) > 1:
                if np.abs(elbo[-1] - elbo[-2]) < tol:
                    print('\n')
                    return elbo
        print('WARNING: meanfield_coordinate_descent hit maxiter of %d' % maxiter)
        return elbo

    @pass_obs_arg
    def meanfield_update(self, obs=None):
        scores, labels = self._meanfield_update_sweep(obs)
        if self.has_data():
            self.labels = labels
        return self.variational_lowerbound(obs, scores)

    def _meanfield_update_sweep(self, obs):
        scores, z = self._meanfield_update_labels(obs)
        self._meanfield_update_parameters(obs, scores)
        return scores, z

    def _meanfield_update_labels(self, obs):
        scores, labels = [], []
        for _obs in obs:
            scores.append(self.expected_scores(_obs))
            labels.append(np.argmax(scores[-1], axis=1))
        return scores, labels

    def _meanfield_update_parameters(self, obs, scores):
        self._meanfield_update_components(obs, scores)
        self._meanfield_update_gating(scores)

    def _meanfield_update_gating(self, scores):
        self.gating.meanfield_update(None, [_score for _score in scores])

    def _meanfield_update_components(self, obs, scores):
        for idx, c in enumerate(self.components):
            c.meanfield_update([_obs for _obs in obs],
                               [_score[:, idx] for _score in scores])

    # SVI
    def meanfield_sgdstep(self, obs, prob, stepsize):
        obs = obs if isinstance(obs, list) else [obs]
        scores, _ = self._meanfield_update_labels(obs)
        self._meanfield_sgdstep_parameters(obs, scores, prob, stepsize)

        if self.has_data():
            for _obs in self.raw_obs:
                self.labels.append(np.argmax(self.scores(_obs), axis=1))

    def _meanfield_sgdstep_parameters(self, obs, scores, prob, stepsize):
        self._meanfield_sgdstep_components(obs, scores, prob, stepsize)
        self._meanfield_sgdstep_gating(scores, prob, stepsize)

    def _meanfield_sgdstep_components(self, obs, scores, prob, stepsize):
        for idx, c in enumerate(self.components):
            c.meanfield_sgdstep([_obs for _obs in obs],
                                [_score[:, idx] for _score in scores], prob, stepsize)

    def _meanfield_sgdstep_gating(self, scores, prob, stepsize):
        self.gating.meanfield_sgdstep(None, [_score for _score in scores], prob, stepsize)

    def _variational_lowerbound_labels(self, scores):
        K = self.size

        # return avg energy plus entropy
        errs = np.seterr(invalid='ignore', divide='ignore')
        prod = scores * np.log(scores)
        prod[np.isnan(prod)] = 0.  # 0 * -inf = 0.
        np.seterr(**errs)

        q_entropy = - prod.sum()

        if isinstance(self.gating, BayesianCategoricalWithDirichlet):
            logpitilde = self.gating.expected_log_likelihood(np.arange(K))
            p_avgengy = (scores * logpitilde).sum()
        elif isinstance(self.gating, BayesianCategoricalWithStickBreaking):
            counts = scores
            cumcounts = np.hstack((np.cumsum(counts[:, ::-1], axis=1)[:, -2::-1],
                                   np.zeros((len(counts), 1))))
            E_log_stick, E_log_rest = self.gating.expected_log_likelihood(np.arange(K))
            p_avgengy = np.sum(cumcounts * E_log_rest + counts * E_log_stick)
        else:
            raise NotImplementedError

        return p_avgengy + q_entropy

    def variational_lowerbound(self, obs, scores):
        vlb = 0.
        vlb += sum(self._variational_lowerbound_labels(_score) for _score in scores)
        vlb += self.gating.variational_lowerbound()
        vlb += sum(c.variational_lowerbound() for c in self.components)
        for _obs, _score in zip(obs, scores):
            vlb += np.sum([r.dot(c.expected_log_likelihood(_obs))
                           for c, r in zip(self.components, _score.T)])

        # add in symmetry factor (if we're actually symmetric)
        if len(set(type(c) for c in self.components)) == 1:
            vlb += special.gammaln(self.size + 1)
        return vlb

    # Misc
    def bic(self, obs=None):
        return -2 * self.log_likelihood(obs) + self.nb_params * np.log(obs.shape[0])

    def aic(self):
        assert self.has_data()
        return 2 * self.nb_params - 2 * sum(self.log_likelihood(_obs) for _obs in self.obs)

    @pass_obs_arg
    def plot(self, obs=None, color=None, legend=False, alpha=None):
        # I haven't implemented plotting
        # for whitend data, it's a hassle :D
        assert self.whitend is False

        import matplotlib.pyplot as plt
        from matplotlib import cm

        artists = []

        # get colors
        cmap = cm.get_cmap()
        if color is None:
            label_colors = dict((idx, cmap(v)) for idx, v in
                                enumerate(np.linspace(0, 1, self.size, endpoint=True)))
        else:
            label_colors = dict((idx, color) for idx in range(self.size))

        labels = []
        for _obs in obs:
            labels.append(np.argmax(self.scores(_obs), axis=1))

        # plot data scatter
        for _obs, _label in zip(obs, labels):
            colorseq = [label_colors[l] for l in _label]
            artists.append(plt.scatter(_obs[:, 0], _obs[:, 1], c=colorseq, s=5))

        # plot parameters
        axis = plt.axis()
        for label, (c, w) in enumerate(zip(self.components, self.gating.probs)):
            artists.extend(c.plot(color=label_colors[label], label='%d' % label,
                                  alpha=min(0.25, 1. - (1. - w) ** 2) / 0.25 if alpha is None else alpha))
        plt.axis(axis)

        # add legend
        if legend and color is None:
            plt.legend([plt.Rectangle((0, 0), 1, 1, fc=c)
                        for i, c in label_colors.items() if i in self.used_labels],
                       [i for i in label_colors if i in self.used_labels], loc='best', ncol=2)
        plt.show()

        return artists


class BayesianMixtureOfLinearGaussians(Conditional):
    """
    This class is for mixtures of other distributions.
    """

    def __init__(self, gating, basis, models):
        assert len(basis) > 0 and len(models) > 0
        assert len(basis) == len(models)

        self.gating = gating
        self.basis = basis  # input density
        self.models = models  # output density

        self.input = []
        self.target = []
        self.labels = []

        self.whitend = False
        self.input_transform = None
        self.target_transform = None

    @property
    def nb_params(self):
        return self.gating.nb_params\
               + sum(b.nb_params for b in self.basis)\
               + sum(m.nb_params for m in self.models)

    @property
    def size(self):
        return len(self.models)

    @property
    def drow(self):
        return self.basis[0].dim()

    @property
    def dcol(self):
        return self.models[0].dcol()

    @property
    def used_labels(self):
        assert self.has_data()
        label_usages = sum(np.bincount(_label, minlength=self.size) for _label in self.labels)
        used_labels, = np.where(label_usages > 0)
        return used_labels

    def add_data(self, y, x, whiten=False):
        y = y if isinstance(y, list) else [y]
        x = x if isinstance(x, list) else [x]
        for _y in y:
            self.labels.append(self.gating.rvs(len(_y)))

        if whiten:
            self.whitend = True
            from sklearn.decomposition import PCA

            Y = np.vstack([_y for _y in y])
            X = np.vstack([_x for _x in x])

            self.target_transform = PCA(n_components=Y.shape[-1], whiten=True)
            self.input_transform = PCA(n_components=X.shape[-1], whiten=True)

            self.target_transform.fit(Y)
            self.input_transform.fit(X)
            for _y, _x in zip(y, x):
                self.target.append(self.target_transform.transform(_y))
                self.input.append(self.input_transform.transform(_x))
        else:
            self.target = y
            self.input = x

    def clear_data(self):
        self.input.clear()
        self.target.clear()
        self.labels.clear()

    def clear_transform(self):
        self.whitend = False
        self.input_transform = None
        self.target_transform = None

    def has_data(self):
        return len(self.target) > 0 and len(self.input) > 0

    def rvs(self, size=1):
        z = self.gating.rvs(size)
        counts = np.bincount(z, minlength=self.size)

        x = np.empty((size, self.dcol))
        y = np.empty((size, self.drow))
        for idx, (b, m, count) in enumerate(zip(self.basis, self.models, counts)):
            x[z == idx, ...] = b.rvs(count)
            y[z == idx, ...] = m.rvs(x[z == idx, ...])

        perm = np.random.permutation(size)
        x, y, z = x[perm], y[perm], z[perm]

        return y, x, z

    def log_likelihood(self, y, x):
        assert isinstance(x, (np.ndarray, list)) and isinstance(y, (np.ndarray, list))
        if isinstance(x, list) and isinstance(y, list):
            return sum(self.log_likelihood(_y, _x) for (_y, _x) in zip(y, x))
        else:
            scores = self.log_scores(y, x)
            idx = np.logical_and(~np.isnan(y).any(1), ~np.isnan(x).any(1))
            return np.sum(logsumexp(scores[idx], axis=1))

    def mean(self, x):
        raise NotImplementedError

    def mode(self, x):
        raise NotImplementedError

    def log_partition(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def log_scores(self, y, x):
        N, K = y.shape[0], self.size

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, (b, m) in enumerate(zip(self.basis, self.models)):
            component_scores[:, idx] = b.log_likelihood(x)
            component_scores[:, idx] += m.log_likelihood(y, x)
        component_scores = np.nan_to_num(component_scores)

        gating_scores = self.gating.log_likelihood(np.arange(K))
        score = gating_scores + component_scores
        return score

    # Expectation-Maximization
    def scores(self, y, x):
        logr = self.log_scores(y, x)
        score = np.exp(logr - np.max(logr, axis=1, keepdims=True))
        score /= np.sum(score, axis=1, keepdims=True)
        return score

    @pass_target_and_input_arg
    def max_likelihood(self, y=None, x=None):
        # Expectation step
        scores = []
        for _y, _x in zip(y, x):
            scores.append(self.scores(_y, _x))

        # Maximization step
        for idx, (b, m) in enumerate(zip(self.basis, self.models)):
            b.max_likelihood([_x for _x in x], [_score[:, idx] for _score in scores])
            m.max_likelihood([_y for _y in y], [_score[:, idx] for _score in scores])

        # mixture weights
        self.gating.max_likelihood(None, [_score for _score in scores])

    # Gibbs sampling
    @pass_target_input_and_labels_arg
    def resample(self, y=None, x=None, z=None):
        self._resample_components(y, x, z)
        self._resample_gating(z)
        z = self._resample_labels(y, x)

        if self.has_data() > 0:
            self.labels = z

    def copy_sample(self):
        new = copy.copy(self)
        new.basis = [b.copy_sample() for b in self.basis]
        new.models = [m.copy_sample() for m in self.models]
        new.gating = self.gating.copy_sample()
        return new

    def _resample_components(self, y, x, z):
        for idx, (b, m) in enumerate(zip(self.basis, self.models)):
            b.resample(data=[_x[_z == idx] for _x, _z in zip(x, z)])
            m.resample(y=[_y[_z == idx] for _y, _z in zip(y, z)],
                       x=[_x[_z == idx] for _x, _z in zip(x, z)])

    def _resample_gating(self, z):
        self.gating.resample([_z for _z in z])

    def _resample_labels(self, y, x):
        z = []
        for _y, _x in zip(y, x):
            score = self.log_scores(_y, _x)
            z.append(sample_discrete_from_log(score, axis=1))
        return z

    # Mean Field
    def expected_scores(self, y, x):
        # compute responsibilities
        N, K = y.shape[0], self.size

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, (b, m) in enumerate(zip(self.basis, self.models)):
            component_scores[:, idx] = b.expected_log_likelihood(x)
            component_scores[:, idx] += m.expected_log_likelihood(y, x)
        component_scores = np.nan_to_num(component_scores)

        if isinstance(self.gating, BayesianCategoricalWithDirichlet):
            gating_scores = self.gating.expected_log_likelihood(np.arange(K))
        elif isinstance(self.gating, BayesianCategoricalWithStickBreaking):
            E_log_stick, E_log_rest = self.gating.expected_log_likelihood(np.arange(K))
            gating_scores = np.take(E_log_stick + np.hstack((0, np.cumsum(E_log_rest)[:-1])), np.arange(K))
        else:
            raise NotImplementedError

        logr = gating_scores + component_scores

        r = np.exp(logr - np.max(logr, axis=1, keepdims=True))
        r /= np.sum(r, axis=1, keepdims=True)

        return r

    def meanfield_coordinate_descent(self, tol=1e-1, maxiter=250, progprint=False):
        elbo = []
        step_iterator = range(maxiter) if not progprint else progprint_xrange(maxiter)
        for _ in step_iterator:
            elbo.append(self.meanfield_update())
            if elbo[-1] is not None and len(elbo) > 1:
                if np.abs(elbo[-1] - elbo[-2]) < tol:
                    print('\n')
                    return elbo
        print('WARNING: meanfield_coordinate_descent hit maxiter of %d' % maxiter)
        return elbo

    @pass_target_and_input_arg
    def meanfield_update(self, y=None, x=None):
        scores, z = self._meanfield_update_sweep(y, x)
        if self.has_data():
            self.labels = z
        return self.variational_lowerbound(y, x, scores)

    def _meanfield_update_sweep(self, y, x):
        scores, z = self._meanfield_update_labels(y, x)
        self._meanfield_update_parameters(y, x, scores)
        return scores, z

    def _meanfield_update_labels(self, y, x):
        scores, z = [], []
        for _y, _x in zip(y, x):
            scores.append(self.expected_scores(_y, _x))
            z.append(np.argmax(scores[-1], axis=1))
        return scores, z

    def _meanfield_update_parameters(self, y, x, scores):
        self._meanfield_update_components(y, x, scores)
        self._meanfield_update_gating(scores)

    def _meanfield_update_gating(self, scores):
        self.gating.meanfield_update(None, [_score for _score in scores])

    def _meanfield_update_components(self, y, x, scores):
        for idx, (b, m) in enumerate(zip(self.basis, self.models)):
            b.meanfield_update([_x for _x in x], [_score[:, idx] for _score in scores])
            m.meanfield_update([_y for _y in y], [_x for _x in x], [_score[:, idx] for _score in scores])

    # SVI
    def meanfield_sgdstep(self, y, x, prob, stepsize):
        y = y if isinstance(y, list) else [y]
        x = x if isinstance(x, list) else [x]

        scores, _ = self._meanfield_update_labels(y, x)
        self._meanfield_sgdstep_parameters(y, x, scores, prob, stepsize)

        if self.has_data():
            for _y, _x in zip(self.raw_target, self.raw_input):
                self.labels.append(np.argmax(self.scores(_y, _x), axis=1))

    def _meanfield_sgdstep_parameters(self, y, x, scores, prob, stepsize):
        self._meanfield_sgdstep_components(y, x, scores, prob, stepsize)
        self._meanfield_sgdstep_gating(scores, prob, stepsize)

    def _meanfield_sgdstep_components(self, y, x, scores, prob, stepsize):
        for idx, (b, m) in enumerate(zip(self.basis, self.models)):
            b.meanfield_sgdstep([_x for _x in x],
                                [_score[:, idx] for _score in scores], prob, stepsize)
            m.meanfield_sgdstep([_y for _y in y], [_x for _x in x],
                                [_score[:, idx] for _score in scores], prob, stepsize)

    def _meanfield_sgdstep_gating(self, scores, prob, stepsize):
        self.gating.meanfield_sgdstep(None, [_score for _score in scores], prob, stepsize)

    def _variational_lowerbound_labels(self, scores):
        K = self.size

        # return avg energy plus entropy
        errs = np.seterr(invalid='ignore', divide='ignore')
        prod = scores * np.log(scores)
        prod[np.isnan(prod)] = 0.  # 0 * -inf = 0.
        np.seterr(**errs)

        q_entropy = - prod.sum()

        if isinstance(self.gating, BayesianCategoricalWithDirichlet):
            logpitilde = self.gating.expected_log_likelihood(np.arange(K))
            p_avgengy = (scores * logpitilde).sum()
        elif isinstance(self.gating, BayesianCategoricalWithStickBreaking):
            counts = scores
            cumcounts = np.hstack((np.cumsum(counts[:, ::-1], axis=1)[:, -2::-1],
                                   np.zeros((len(counts), 1))))
            E_log_stick, E_log_rest = self.gating.expected_log_likelihood(np.arange(K))
            p_avgengy = np.sum(cumcounts * E_log_rest + counts * E_log_stick)
        else:
            raise NotImplementedError

        return p_avgengy + q_entropy

    def variational_lowerbound(self, y, x, scores):
        vlb = 0.
        vlb += sum(self._variational_lowerbound_labels(_score) for _score in scores)
        vlb += self.gating.variational_lowerbound()
        vlb += sum(b.variational_lowerbound() for b in self.basis)
        vlb += sum(m.variational_lowerbound() for m in self.models)
        for _y, _x, _score in zip(y, x, scores):
            vlb += np.sum([r.dot(b.expected_log_likelihood(_x)) for b, r in zip(self.basis, _score.T)])
            vlb += np.sum([r.dot(m.expected_log_likelihood(_y, _x)) for m, r in zip(self.models, _score.T)])

        # add in symmetry factor (if we're actually symmetric)
        if len(set(type(m) for m in self.models)) == 1:
            vlb += special.gammaln(self.size + 1)
        return vlb

    # Misc
    def bic(self, y=None, x=None):
        return -2 * self.log_likelihood(y, x) + self.nb_params * np.log(y.shape[0])

    def aic(self):
        assert self.has_data()
        return 2 * self.nb_params - 2 * sum(self.log_likelihood(_y, _x)
                                            for _y, _x in zip(self.target, self.input))

    def meanfield_predictive_activation(self, x, sparse=True, type='gaussian'):
        x = np.atleast_2d(x).T

        _x = x if not self.whitend\
            else self.input_transform.transform(x)

        _labels = self.used_labels if sparse else range(self.size)

        activations = np.zeros((len(_x), len(_labels)))
        for n in range(len(_x)):
            for i, idx in enumerate(_labels):
                if type == 'gaussian':
                    _act = self.basis[idx].log_posterior_predictive_gaussian(_x[n, :])
                else:
                    _act = self.basis[idx].log_posterior_predictive_studentt(_x[n, :])
                activations[n, i] = np.exp(_act)

        activations = activations / np.sum(activations, axis=1, keepdims=True)
        return activations

    def meanfield_predictive_gating(self, x, sparse=False, type='gaussian'):
        # compute posterior mixing weights
        weights = self.gating.posterior.mean()

        # calculate the marginal likelihood of query for each cluster
        # calculate the normalization term for mean function for query
        marginal_likelihood = np.zeros((self.size, ))
        effective_weights = np.zeros((self.size, ))

        _labels = self.used_labels if sparse else range(self.size)
        for idx in _labels:
            if type == 'gaussian':
                marginal_likelihood[idx] = np.exp(self.basis[idx].log_posterior_predictive_gaussian(x))
            elif type == 'studentt':
                marginal_likelihood[idx] = np.exp(self.basis[idx].log_posterior_predictive_studentt(x))
            effective_weights[idx] = weights[idx] * marginal_likelihood[idx]

        effective_weights = effective_weights / np.sum(effective_weights)
        return effective_weights

    def meanfield_prediction(self, x, y=None, compute_nlpd=False,
                             prediction='average', incremental=False,
                             type='gaussian', sparse=False):

        if compute_nlpd:
            assert y is not None

        if self.whitend:
            x = np.squeeze(self.input_transform.transform(np.atleast_2d(x)))
            if y is not None:
                y = np.squeeze(self.target_transform.transform(np.atleast_2d(y)))

        target, input = y, x

        dim = self.models[0].drow
        mu, var, stdv, nlpd = np.zeros((dim, )), np.zeros((dim, )), np.zeros((dim, )), 0.

        weights = self.meanfield_predictive_gating(input, sparse, type)

        if prediction == 'mode':
            mode = np.argmax(weights)

            if type == 'gaussian':
                mu, _sigma, df = self.models[mode].predictive_posterior_gaussian(input)
                var = np.diag(_sigma)  # consider only diagonal variances for plots
                if compute_nlpd:
                    nlpd = np.exp(mvn_logpdf(target, mu, _sigma))
            else:
                mu, _sigma, df = self.models[mode].predictive_posterior_studentt(input)
                var = np.diag(_sigma * df / (df - 2))  # consider only diagonal variances for plots
                if compute_nlpd:
                    nlpd = np.exp(mvt_logpdf(target, mu, _sigma, df))

        elif prediction == 'average':
            _labels = self.used_labels if sparse else range(self.size)
            for idx in _labels:
                if type == 'gaussian':
                    _mu, _sigma, _df = self.models[idx].predictive_posterior_gaussian(input)
                    _var = np.diag(_sigma)  # consider only diagonal variances for plots
                    if compute_nlpd:
                        nlpd += weights[idx] * np.exp(mvn_logpdf(target, _mu, _sigma))
                else:
                    _mu, _sigma, _df = self.models[idx].predictive_posterior_studentt(input)
                    _var = np.diag(_sigma * _df / (_df - 2))  # consider only diagonal variances for plots
                    if compute_nlpd:
                        nlpd += weights[idx] * np.exp(mvt_logpdf(target, _mu, _sigma, _df))

                # Mean of a mixture = sum of weighted means
                mu += _mu * weights[idx]

                # Variance of a mixture = sum of weighted variances + ...
                # ... + sum of weighted squared means - squared sum of weighted means
                var += (_var + _mu**2) * weights[idx]
            var -= mu**2

        nlpd = - 1.0 * np.log(nlpd) if compute_nlpd else None

        if self.whitend:
            mu = np.squeeze(self.target_transform.inverse_transform(np.atleast_2d(mu)))
            trans = (np.sqrt(self.target_transform.explained_variance_[:, None])
                     * self.target_transform.components_).T
            var = np.diag(trans.T @ np.diag(var) @ trans)

        # only diagonal elements
        stdv = np.sqrt(var)

        if incremental:
            mu += x[:dim]

        return mu, var, stdv, nlpd

    def parallel_meanfield_prediction(self, x, y=None, compute_nlpd=False,
                                      prediction='average', incremental=False,
                                      type='gaussian', sparse=False):

        def _loop(kwargs):
            return self.meanfield_prediction(kwargs['x'], kwargs['y'],
                                             compute_nlpd, prediction,
                                             incremental, type, sparse)

        kwargs_list = []
        for n in range(len(x)):
            _x = x[n, :]
            _y = None if y is None else y[n, :]
            kwargs_list.append({'x': _x, 'y': _y})

        with Pool(processes=nb_cores) as p:
            res = p.map(_loop, kwargs_list, chunksize=int(len(x) / nb_cores))

        mean, var, std, nlpd = list(map(np.vstack, zip(*res)))
        return mean, var, std, nlpd