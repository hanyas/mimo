import numpy as np
from scipy import special as special
from scipy.special import logsumexp

from mimo.abstraction import Conditional

from mimo.distributions.bayesian import CategoricalWithDirichlet
from mimo.distributions.bayesian import CategoricalWithStickBreaking

from mimo.util.decorate import pass_target_and_input_arg
from mimo.util.decorate import pass_target_input_and_labels_arg

from mimo.util.stats import sample_discrete_from_log

from mimo.util.text import progprint_xrange

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

eps = np.finfo(np.float64).tiny


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
        return self.gating.likelihood.nb_params\
               + sum(b.likelihood.nb_params for b in self.basis)\
               + sum(m.likelihood.nb_params for m in self.models)

    @property
    def size(self):
        return len(self.models)

    @property
    def drow(self):
        return self.models[0].likelihood.drow

    @property
    def dcol(self):
        return self.models[0].likelihood.dcol

    @property
    def used_labels(self):
        assert self.has_data()
        label_usages = sum(np.bincount(_label, minlength=self.size)
                           for _label in self.labels)
        used_labels, = np.where(label_usages > 0)
        return used_labels

    def add_data(self, y, x, whiten=False,
                 target_transform=False,
                 input_transform=False,
                 transform_type='PCA'):

        y = y if isinstance(y, list) else [y]
        x = x if isinstance(x, list) else [x]
        for _y in y:
            self.labels.append(self.gating.likelihood.rvs(len(_y)))

        if whiten:
            self.whitend = True

            if not (target_transform and input_transform):

                Y = np.vstack([_y for _y in y])
                X = np.vstack([_x for _x in x])

                if transform_type == 'PCA':
                    self.target_transform = PCA(n_components=Y.shape[-1], whiten=True)
                    self.input_transform = PCA(n_components=X.shape[-1], whiten=True)
                else:
                    self.target_transform = StandardScaler()
                    self.input_transform = StandardScaler()

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
        z = self.gating.likelihood.rvs(size)
        counts = np.bincount(z, minlength=self.size)

        x = np.empty((size, self.dcol))
        y = np.empty((size, self.drow))
        for idx, (b, m, count) in enumerate(zip(self.basis, self.models, counts)):
            x[z == idx, ...] = b.likelihood.rvs(count)
            y[z == idx, ...] = m.likelihood.rvs(x[z == idx, ...])

        perm = np.random.permutation(size)
        x, y, z = x[perm], y[perm], z[perm]

        return y, x, z

    def log_likelihood(self, y, x):
        assert isinstance(x, (np.ndarray, list)) and isinstance(y, (np.ndarray, list))
        if isinstance(x, list) and isinstance(y, list):
            return sum(self.log_likelihood(_y, _x) for (_y, _x) in zip(y, x))
        else:
            scores = self.log_scores(y, x)
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
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
            component_scores[:, idx] = b.likelihood.log_likelihood(x)
            component_scores[:, idx] += m.likelihood.log_likelihood(y, x)
        component_scores = np.nan_to_num(component_scores)

        gating_scores = self.gating.likelihood.log_likelihood(np.arange(K))
        score = gating_scores + component_scores
        return score

    # Expectation-Maximization
    def scores(self, y, x):
        logr = self.log_scores(y, x)
        score = np.exp(logr - np.max(logr, axis=1, keepdims=True))
        score /= np.sum(score, axis=1, keepdims=True)
        return score

    @pass_target_and_input_arg
    def max_aposteriori(self, y=None, x=None):
        # Expectation step
        scores = []
        for _y, _x in zip(y, x):
            scores.append(self.scores(_y, _x))

        # Maximization step
        for idx, (b, m) in enumerate(zip(self.basis, self.models)):
            b.max_aposteriori([_x for _x in x], [_score[:, idx] for _score in scores])
            m.max_aposteriori([_y for _y in y], [_score[:, idx] for _score in scores])

        # mixture weights
        self.gating.max_aposteriori(None, scores)

    # Gibbs sampling
    @pass_target_input_and_labels_arg
    def resample(self, y=None, x=None, z=None):
        self._resample_components(y, x, z)
        self._resample_gating(z)
        z = self._resample_labels(y, x)

        if self.has_data():
            self.labels = z

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
        N, K = y.shape[0], self.size

        component_scores = np.empty((N, K))
        for idx, (b, m) in enumerate(zip(self.basis, self.models)):
            component_scores[:, idx] = b.posterior.expected_log_likelihood(x)
            component_scores[:, idx] += m.posterior.expected_log_likelihood(y, x, m.likelihood.affine)
        component_scores = np.nan_to_num(component_scores)

        if isinstance(self.gating, CategoricalWithDirichlet):
            gating_scores = self.gating.posterior.expected_statistics()
        elif isinstance(self.gating, CategoricalWithStickBreaking):
            E_log_stick, E_log_rest = self.gating.posterior.expected_statistics()
            gating_scores = E_log_stick + np.hstack((0, np.cumsum(E_log_rest)[:-1]))
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
        # return self.variational_lowerbound(y, x, scores)

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
            for _y, _x in zip(self.target, self.input):
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
        self.gating.meanfield_sgdstep(None, scores, prob, stepsize)

    def _variational_lowerbound_labels(self, scores):
        vlb = 0.

        if isinstance(self.gating, CategoricalWithDirichlet):
            vlb += np.sum(scores * self.gating.posterior.expected_log_likelihood())
        elif isinstance(self.gating, CategoricalWithStickBreaking):
            cumscores = np.hstack((np.cumsum(scores[:, ::-1], axis=1)[:, -2::-1],
                                   np.zeros((len(scores), 1))))
            E_log_stick, E_log_rest = self.gating.posterior.expected_log_likelihood()
            vlb += np.sum(scores * E_log_stick + cumscores * E_log_rest)

        errs = np.seterr(invalid='ignore', divide='ignore')
        vlb -= np.nansum(scores * np.log(scores))  # treats nans as zeros
        np.seterr(**errs)

        return vlb

    def _variational_lowerbound_data(self, y, x, scores):
        vlb = 0.
        vlb += np.sum([r.dot(b.posterior.expected_log_likelihood(x)) for b, r in zip(self.basis, scores.T)])
        vlb += np.sum([r.dot(m.posterior.expected_log_likelihood(y, x, m.likelihood.affine)) for m, r in zip(self.models, scores.T)])
        return vlb

    def variational_lowerbound(self, y, x, scores):
        vlb = 0.
        vlb += sum(self._variational_lowerbound_labels(_score) for _score in scores)
        vlb += self.gating.variational_lowerbound()
        vlb += sum(b.variational_lowerbound() for b in self.basis)
        vlb += sum(m.variational_lowerbound() for m in self.models)
        for _y, _x, _score in zip(y, x, scores):
            vlb += self._variational_lowerbound_data(_y, _x, _score)

        # add in symmetry factor (if we're actually symmetric)
        if len(set(type(m) for m in self.models)) == 1:
            vlb += special.gammaln(self.size + 1)
        return vlb

    # Misc
    def bic(self, y=None, x=None):
        assert x is not None and y is not None
        return - 2. * self.log_likelihood(y, x) + self.nb_params\
               * np.log(sum([_y.shape[0] for _y in y]))

    def aic(self):
        assert self.has_data()
        return 2. * self.nb_params - 2. * sum(self.log_likelihood(_y, _x)
                                              for _y, _x in zip(self.target, self.input))

    def meanfield_predictive_activation(self, x, dist='gaussian'):
        # Mainly for plotting basis functions
        x = np.reshape(x, (-1, self.dcol))

        x = x if not self.whitend \
            else self.input_transform.transform(x)

        labels = self.used_labels
        activations = np.zeros((len(x), len(labels)))

        for i, idx in enumerate(labels):
            activations[:, i] = self.basis[idx].log_posterior_predictive_gaussian(x)\
                if dist == 'gaussian' else self.basis[idx].log_posterior_predictive_studentt(x)

        activations = np.exp(activations)
        activations = activations / np.sum(activations, axis=1, keepdims=True)
        return activations

    def meanfield_predictive_gating(self, x, dist='gaussian'):
        # compute posterior mixing weights
        weights = self.gating.posterior.mean()

        labels = range(self.size)
        log_posterior_predictive = np.zeros((len(x), len(labels)))

        for i, idx in enumerate(labels):
            log_posterior_predictive[:, i] = self.basis[idx].log_posterior_predictive_gaussian(x)\
                if dist == 'gaussian' else self.basis[idx].log_posterior_predictive_studentt(x)

        effective_weights = weights[labels] * np.exp(log_posterior_predictive)
        effective_weights = effective_weights / np.sum(effective_weights, axis=1, keepdims=True)
        return effective_weights

    def meanfield_predictive_moments(self, x, dist='gaussian'):
        # returns only diagonal variance
        mu, var = np.zeros((len(x), self.drow, self.size)),\
                  np.zeros((len(x), self.drow, self.size))

        for n, model in enumerate(self.models):
            if dist == 'gaussian':
                mu[..., n], _lmbda = model.posterior_predictive_gaussian(x)
                var[..., n] = 1. / np.vstack(list(map(np.diag, _lmbda)))
            else:
                mu[..., n], _lmbda, _df = model.posterior_predictive_studentt(x)
                var[..., n] = _df / (_df - 2) * (1. / np.vstack(list(map(np.diag, _lmbda))))

        return mu, var

    def meanfiled_log_predictive_likelihood(self, y, x, dist='gaussian'):
        lpd = np.zeros((len(x), self.size))

        for n, model in enumerate(self.models):
            lpd[:, n] = model.log_posterior_predictive_gaussian(y, x)\
                if dist == 'gaussian' else model.log_posterior_predictive_studentt(y, x)

        return lpd

    @staticmethod
    def _mixture_moments(mus, vars, weights):
        # Mean of a mixture = sum of weighted means
        mu = np.einsum('nkl,nl->nk', mus, weights)
        # Variance of a mixture = sum of weighted variances + ...
        # ... + sum of weighted squared means - squared sum of weighted means
        var = np.einsum('nkl,nl->nk', vars + mus ** 2, weights) - mu ** 2
        return mu, var

    def meanfield_eleatoric(self, dist='gaussian'):
        var = np.zeros((self.drow, self.drow, self.size))
        for n, model in enumerate(self.models):
            _, _, psi, nu = model.posterior.params
            df = nu - model.likelihood.drow + 1
            var[..., n] = np.linalg.inv(psi * df) if dist == 'gaussian'\
                else np.linalg.inv(psi * df) * df / (df - 2.)

        return var

    def meanfield_prediction(self, x, y=None,
                             prediction='average',
                             incremental=False,
                             dist='gaussian'):

        x = np.reshape(x, (-1, self.dcol))

        compute_nlpd = False
        if y is not None:
            y = np.reshape(y, (-1, self.drow))
            compute_nlpd = True

        from mimo.util.data import transform, inverse_transform

        input = transform(x, trans=self.input_transform)
        target = None if y is None else transform(y, trans=self.target_transform)

        weights = self.meanfield_predictive_gating(input, dist)
        mus, vars = self.meanfield_predictive_moments(input, dist)

        if prediction == 'mode':
            k = np.argmax(weights, axis=1)
            idx = (range(len(k)), ..., k)
            mu, var = mus[idx], vars[idx]
        elif prediction == 'average':
            labels = range(self.size)
            mu, var = self._mixture_moments(mus[..., labels],
                                            vars[..., labels], weights)
        else:
            raise NotImplementedError

        nlpd = None
        if compute_nlpd:
            lpd = self.meanfiled_log_predictive_likelihood(target, input)
            lw = np.log(weights + eps)
            nlpd = -1.0 * logsumexp(lpd + lw, axis=1)

        mu, var = inverse_transform(mu, var, trans=self.target_transform)

        if incremental:
            mu += x[:, :self.drow]

        if compute_nlpd:
            return mu, var, np.sqrt(var), nlpd
        else:
            return mu, var, np.sqrt(var)
