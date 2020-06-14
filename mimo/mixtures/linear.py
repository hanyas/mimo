import copy
from multiprocessing.pool import Pool as Pool

import numpy as np
from scipy import special as special
from scipy.special import logsumexp

from mimo.abstraction import Conditional

from mimo.distributions.bayesian import CategoricalWithDirichlet
from mimo.distributions.bayesian import CategoricalWithStickBreaking

from mimo.util.decorate import pass_target_and_input_arg
from mimo.util.decorate import pass_target_input_and_labels_arg

from mimo.util.stats import sample_discrete_from_log
from mimo.util.stats import multivariate_gaussian_loglik as mvn_logpdf
from mimo.util.stats import multivariate_studentt_loglik as mvt_logpdf

from mimo.util.text import progprint_xrange

import pathos
nb_cores = pathos.multiprocessing.cpu_count()


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

    def add_data(self, y, x, whiten=False,
                 target_transform=False,
                 input_transform=False):

        y = y if isinstance(y, list) else [y]
        x = x if isinstance(x, list) else [x]
        for _y in y:
            self.labels.append(self.gating.rvs(len(_y)))

        if whiten:
            self.whitend = True

            if not (target_transform and input_transform):
                from sklearn.decomposition import PCA

                Y = np.vstack([_y for _y in y])
                X = np.vstack([_x for _x in x])

                self.target_transform = PCA(n_components=Y.shape[-1], whiten=True)
                self.input_transform = PCA(n_components=X.shape[-1], whiten=True)

                self.target_transform.fit(Y)
                self.input_transform.fit(X)
            else:
                self.target_transform = target_transform
                self.input_transform = input_transform

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

        if isinstance(self.gating, CategoricalWithDirichlet):
            gating_scores = self.gating.expected_log_likelihood(np.arange(K))
        elif isinstance(self.gating, CategoricalWithStickBreaking):
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
                    if progprint:
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

        if isinstance(self.gating, CategoricalWithDirichlet):
            logpitilde = self.gating.expected_log_likelihood(np.arange(K))
            p_avgengy = (scores * logpitilde).sum()
        elif isinstance(self.gating, CategoricalWithStickBreaking):
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
        x = np.atleast_2d(x).T if x.ndim < 2 else x

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