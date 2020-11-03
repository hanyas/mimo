import numpy as np
from scipy import special as special
from scipy.special import logsumexp

from mimo.abstraction import ConditionalMixtureDistribution
from mimo.abstraction import BayesianConditionalMixtureDistribution

from mimo.distributions.bayesian import CategoricalWithDirichlet
from mimo.distributions.bayesian import CategoricalWithStickBreaking

from mimo.util.decorate import pass_target_and_input_arg
from mimo.util.decorate import pass_target_input_and_labels_arg

from mimo.util.stats import sample_discrete_from_log
from mimo.util.data import batches

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm
from pathos.helpers import mp

import pathos
from pathos.pools import ThreadPool as Pool
nb_cores = pathos.multiprocessing.cpu_count()

eps = np.finfo(np.float64).tiny


class MixtureOfLinearGaussians(ConditionalMixtureDistribution):
    """
    This class is for mixtures of Linear Gaussians.
    """
    def __init__(self, gating, basis, models):
        assert len(basis) > 0 and len(models) > 0
        assert len(basis) == len(models)

        self.gating = gating
        self.basis = basis  # input density
        self.models = models  # output density

    @property
    def params(self):
        raise NotImplementedError

    @property
    def nb_params(self):
        return sum(b.nb_params for b in self.basis) + self.gating.nb_params\
               + sum(m.nb_params for m in self.models)

    @property
    def size(self):
        return len(self.basis)

    @property
    def drow(self):
        return self.models[0].drow

    @property
    def dcol(self):
        return self.models[0].dcol

    @property
    def dim(self):
        return self.drow, self.dcol

    def rvs(self, size=1):
        z = self.gating.likelihood.rvs(size)
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
            return [self.log_likelihood(_y, _x) for (_y, _x) in zip(y, x)]
        else:
            scores = self.log_scores(y, x)
            idx = np.logical_and(~np.isnan(y).any(axis=1),
                                 ~np.isnan(x).any(axis=1))
            return logsumexp(scores[idx], axis=1)

    def log_scores(self, y, x):
        N, K = y.shape[0], self.size

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, (b, m) in enumerate(zip(self.basis, self.models)):
            component_scores[:, idx] = b.log_likelihood(x)
            component_scores[:, idx] += m.log_likelihood(y, x)
        component_scores = np.nan_to_num(component_scores, copy=False)

        gating_scores = self.gating.log_likelihood(np.arange(K))
        score = gating_scores + component_scores
        return score

    # Expectation-Maximization
    def scores(self, y, x):
        logr = self.log_scores(y, x)
        score = np.exp(logr - np.max(logr, axis=1, keepdims=True))
        score /= np.sum(score, axis=1, keepdims=True)
        return score

    def max_likelihood(self, y=None, x=None, maxiter=1, progprint=True):

        current = mp.current_process()
        if len(current._identity) > 0:
            pos = current._identity[0] - 1
        else:
            pos = 0

        y = y if isinstance(y, list) else [y]
        x = x if isinstance(y, list) else [x]

        elbo = []
        with tqdm(total=maxiter, desc=f'MAP #{pos + 1}',
                  position=pos, disable=not progprint) as pbar:
            for i in range(maxiter):
                # Expectation step
                scores = []
                for _y, _x in zip(y, x):
                    scores.append(self.scores(_y, _x))

                # Maximization step
                for idx, (b, m) in enumerate(zip(self.basis, self.models)):
                    b.max_likelihood([_x for _x in x], [_score[:, idx] for _score in scores])
                    m.max_likelihood([_y for _y in y], [_score[:, idx] for _score in scores])

                # mixture weights
                self.gating.max_likelihood(None, scores)

                elbo.append(np.sum(self.log_likelihood(y, x)))
                pbar.update(1)

        return elbo


class BayesianMixtureOfLinearGaussians(BayesianConditionalMixtureDistribution):
    """
    This class is for Bayesian mixtures of Linear Gaussians.
    """

    def __init__(self, gating, basis, models):
        assert len(basis) > 0 and len(models) > 0
        assert len(basis) == len(models)

        self.gating = gating
        self.basis = basis  # input density
        self.models = models  # output density

        self.likelihood = MixtureOfLinearGaussians(gating=self.gating.likelihood,
                                                   basis=[b.likelihood for b in self.basis],
                                                   models=[m.likelihood for m in self.models])

        self.input = []
        self.target = []
        self.labels = []

        self.whitend = False
        self.input_transform = None
        self.target_transform = None

    @property
    def used_labels(self):
        assert self.has_data()
        label_usages = sum(np.bincount(_label, minlength=self.likelihood.size)
                           for _label in self.labels)
        used_labels, = np.where(label_usages > 0)
        return used_labels

    def add_data(self, y, x=None, whiten=False,
                 target_transform=False,
                 input_transform=False,
                 transform_type='PCA',
                 labels_from_prior=False):

        y = y if isinstance(y, list) else [y]
        x = x if isinstance(x, list) else [x]

        if whiten:
            self.whitend = True

            if not (target_transform and input_transform):

                Y = np.vstack([_y for _y in y])
                X = np.vstack([_x for _x in x])

                if transform_type == 'PCA':
                    self.target_transform = PCA(n_components=Y.shape[-1], whiten=True)
                    self.input_transform = PCA(n_components=X.shape[-1], whiten=True)
                elif transform_type == 'Standard':
                    self.target_transform = StandardScaler()
                    self.input_transform = StandardScaler()
                elif transform_type == 'MinMax':
                    self.target_transform = MinMaxScaler((-1., 1.))
                    self.input_transform = MinMaxScaler((-1., 1.))
                else:
                    raise NotImplementedError

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

        if labels_from_prior:
            for _y, _x in zip(self.target, self.input):
                self.labels.append(self.likelihood.gating.rvs(len(_y)))
        else:
            self.labels = self._resample_labels(self.target, self.input)

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

    @pass_target_and_input_arg
    def max_aposteriori(self, y=None, x=None, maxiter=1, progprint=True):

        current = mp.current_process()
        if len(current._identity) > 0:
            pos = current._identity[0] - 1
        else:
            pos = 0

        with tqdm(total=maxiter, desc=f'MAP #{pos + 1}',
                  position=pos, disable=not progprint) as pbar:
            for i in range(maxiter):
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

                pbar.update(1)

    # Gibbs sampling
    @pass_target_input_and_labels_arg
    def resample(self, y=None, x=None, z=None,
                 maxiter=1, progprint=True):

        current = mp.current_process()
        if len(current._identity) > 0:
            pos = current._identity[0] - 1
        else:
            pos = 0

        with tqdm(total=maxiter, desc=f'Gibbs #{pos + 1}',
                  position=pos, disable=not progprint) as pbar:
            for _ in range(maxiter):
                self._resample_components(y, x, z)
                self._resample_gating(z)
                z = self._resample_labels(y, x)

                if self.has_data():
                    self.labels = z

                pbar.update(1)

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
            score = self.likelihood.log_scores(_y, _x)
            z.append(sample_discrete_from_log(score, axis=1))
        return z

    # Mean Field
    def expected_scores(self, y, x, nb_threads=4):
        N, K = y.shape[0], self.likelihood.size

        component_scores = np.empty((N, K))

        if nb_threads == 1:
            for idx, (b, m) in enumerate(zip(self.basis, self.models)):
                _affine = m.likelihood.affine
                component_scores[:, idx] = b.posterior.expected_log_likelihood(x)
                component_scores[:, idx] += m.posterior.expected_log_likelihood(y, x, _affine)
        else:
            def _loop(idx):
                _affine = self.models[idx].likelihood.affine
                component_scores[:, idx] = self.basis[idx].posterior.expected_log_likelihood(x)
                component_scores[:, idx] += self.models[idx].posterior.expected_log_likelihood(y, x, _affine)

            with Pool(threads=nb_threads) as p:
                p.map(_loop, range(self.likelihood.size))

        component_scores = np.nan_to_num(component_scores, copy=False)

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

    def meanfield_coordinate_descent(self, tol=1e-2, maxiter=250, progprint=True):
        elbo = []

        current = mp.current_process()
        if len(current._identity) > 0:
            pos = current._identity[0] - 1
        else:
            pos = 0

        with tqdm(total=maxiter, desc=f'VI #{pos + 1}',
                  position=pos, disable=not progprint) as pbar:
            for i in range(maxiter):
                elbo.append(self.meanfield_update())
                if elbo[-1] is not None and len(elbo) > 1:
                    if elbo[-1] < elbo[-2]:
                        print('WARNING: ELBO should always increase')
                        return elbo
                    if (elbo[-1] - elbo[-2]) < tol:
                        return elbo
                pbar.update(1)

        # print('WARNING: meanfield_coordinate_descent hit maxiter of %d' % maxiter)
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
        self.gating.meanfield_update(None, scores)

    def _meanfield_update_components(self, y, x, scores,
                                     nb_threads=1):
        if nb_threads == 1:
            for idx, (b, m) in enumerate(zip(self.basis, self.models)):
                b.meanfield_update(x, [_score[:, idx] for _score in scores])
                m.meanfield_update(y, x, [_score[:, idx] for _score in scores])
        else:
            def _loop(idx):
                self.basis[idx].meanfield_update(x, [_score[:, idx] for _score in scores])
                self.models[idx].meanfield_update(y, x, [_score[:, idx] for _score in scores])

            with Pool(threads=nb_threads) as p:
                p.map(_loop, range(self.likelihood.size))

    # SVI
    def meanfield_stochastic_descent(self, stepsize=1e-3, batchsize=128,
                                     maxiter=500, progprint=True):

        assert self.has_data()

        current = mp.current_process()
        if len(current._identity) > 0:
            pos = current._identity[0] - 1
        else:
            pos = 0

        x, y = self.input, self.target
        prob = batchsize / float(sum(len(_x) for _x in x))

        with tqdm(total=maxiter, desc=f'SVI #{pos + 1}',
                  position=pos, disable=not progprint) as pbar:
            for _ in range(maxiter):
                for _x, _y in zip(x, y):
                    for batch in batches(batchsize, len(_x)):
                        _mx, _my = _x[batch, :], _y[batch, :]
                        self.meanfield_sgdstep(_my, _mx, prob, stepsize)
                pbar.update(1)

    def meanfield_sgdstep(self, y, x, prob, stepsize):
        y = y if isinstance(y, list) else [y]
        x = x if isinstance(x, list) else [x]

        scores, _ = self._meanfield_update_labels(y, x)
        self._meanfield_sgdstep_parameters(y, x, scores, prob, stepsize)

        if self.has_data():
            for _y, _x in zip(self.target, self.input):
                self.labels.append(np.argmax(self.expected_scores(_y, _x), axis=1))

    def _meanfield_sgdstep_parameters(self, y, x, scores, prob, stepsize):
        self._meanfield_sgdstep_components(y, x, scores, prob, stepsize)
        self._meanfield_sgdstep_gating(scores, prob, stepsize)

    def _meanfield_sgdstep_components(self, y, x, scores, prob,
                                      stepsize, nb_threads=4):

        if nb_threads == 1:
            for idx, (b, m) in enumerate(zip(self.basis, self.models)):
                b.meanfield_sgdstep(x, [_score[:, idx] for _score in scores], prob, stepsize)
                m.meanfield_sgdstep(y, x, [_score[:, idx] for _score in scores], prob, stepsize)
        else:
            def _loop(idx):
                self.basis[idx].meanfield_sgdstep(x, [_score[:, idx] for _score in scores], prob, stepsize)
                self.models[idx].meanfield_sgdstep(y, x, [_score[:, idx] for _score in scores], prob, stepsize)

            with Pool(threads=nb_threads) as p:
                p.map(_loop, range(self.likelihood.size))

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
        vlb += np.sum([r.dot(b.posterior.expected_log_likelihood(x))
                       for b, r in zip(self.basis, scores.T)])
        vlb += np.sum([r.dot(m.posterior.expected_log_likelihood(y, x, m.likelihood.affine))
                       for m, r in zip(self.models, scores.T)])
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
            vlb += special.gammaln(self.likelihood.size + 1)
        return vlb

    # Misc
    def bic(self, y=None, x=None):
        assert x is not None and y is not None
        return - 2. * np.sum(self.likelihood.log_likelihood(y, x)) + self.likelihood.nb_params\
               * np.log(sum([_y.shape[0] for _y in y]))

    def aic(self):
        assert self.has_data()
        return 2. * self.likelihood.nb_params - 2. * sum(np.sum(self.likelihood.log_likelihood(_y, _x))
                                                         for _y, _x in zip(self.target, self.input))

    def meanfield_predictive_activation(self, x, dist='gaussian'):
        # Mainly for plotting basis functions
        x = np.reshape(x, (-1, self.likelihood.dcol))

        x = x if not self.whitend \
            else self.input_transform.transform(x)

        weights = self.gating.posterior.mean()

        labels = range(self.likelihood.size)
        activations = np.zeros((len(x), len(labels)))

        for i, idx in enumerate(labels):
            activations[:, i] = self.basis[idx].log_posterior_predictive_gaussian(x)\
                if dist == 'gaussian' else self.basis[idx].log_posterior_predictive_studentt(x)

        activations = weights[labels] * np.exp(activations) + eps
        activations = activations / np.sum(activations, axis=1, keepdims=True)
        return activations

    def meanfield_predictive_gating(self, x, dist='gaussian'):
        # compute posterior mixing weights
        weights = self.gating.posterior.mean()

        labels = range(self.likelihood.size)
        log_posterior_predictive = np.zeros((len(x), len(labels)))

        for i, idx in enumerate(labels):
            log_posterior_predictive[:, i] = self.basis[idx].log_posterior_predictive_gaussian(x)\
                if dist == 'gaussian' else self.basis[idx].log_posterior_predictive_studentt(x)

        effective_weights = weights[labels] * np.exp(log_posterior_predictive) + eps
        effective_weights = effective_weights / np.sum(effective_weights, axis=1, keepdims=True)
        return effective_weights

    def meanfield_predictive_moments(self, x, dist='gaussian', aleatoric_only=False):
        mu, var = np.zeros((len(x), self.likelihood.drow, self.likelihood.size)),\
                  np.zeros((len(x), self.likelihood.drow, self.likelihood.drow, self.likelihood.size))

        for n, model in enumerate(self.models):
            if dist == 'gaussian':
                mu[..., n], _lmbda = model.posterior_predictive_gaussian(x, aleatoric_only)
                var[..., n] = np.linalg.inv(_lmbda)
            else:
                mu[..., n], _lmbda, _df = model.posterior_predictive_studentt(x, aleatoric_only)
                var[..., n] = np.linalg.inv(_lmbda) * _df / (_df - 2)

        return mu, var

    def meanfiled_log_predictive_likelihood(self, y, x, dist='gaussian'):
        lpd = np.zeros((len(x), self.likelihood.size))

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
        var = np.einsum('nkhl,nl->nkh', vars + np.einsum('nkl,nhl->nkhl', mus, mus), weights)\
              - np.einsum('nk,nh->nkh', mu, mu)
        return mu, var

    def meanfield_predictive_aleatoric(self, dist='gaussian'):
        from mimo.util.data import inverse_transform_variance
        weights = self.gating.posterior.mean()

        mus, vars = np.zeros((self.likelihood.size, self.likelihood.drow)),\
                    np.zeros((self.likelihood.size, self.likelihood.drow, self.likelihood.drow))

        for n, (basis, model) in enumerate(zip(self.basis, self.models)):
            x = basis.posterior.gaussian.mu
            if dist == 'gaussian':
                mus[n, :], _lmbda = model.posterior_predictive_gaussian(x, True)
                vars[n, ...] = np.linalg.inv(_lmbda)
            else:
                mus[n, :], _lmbda, _df = model.posterior_predictive_studentt(x, True)
                vars[n, ...] = np.linalg.inv(_lmbda) * _df / (_df - 2)

        mu = np.einsum('nk,n->k', mus, weights)
        var = np.einsum('nkh,n->kh', vars + np.einsum('nk,nh->nkh', mus, mus), weights)\
              - np.einsum('k,h->kh', mu, mu)

        return inverse_transform_variance(var, self.target_transform)

    def meanfield_prediction(self, x, y=None,
                             prediction='average',
                             dist='gaussian',
                             incremental=False,
                             variance='diagonal'):

        x = np.reshape(x, (-1, self.likelihood.dcol))

        compute_nlpd = False
        if y is not None:
            y = np.reshape(y, (-1, self.likelihood.drow))
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
            labels = range(self.likelihood.size)
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
            mu += x[:, :self.likelihood.drow]

        diag = np.vstack(list(map(np.diag, var)))

        if compute_nlpd:
            if variance == 'diagonal':
                return mu, diag, np.sqrt(diag), nlpd
            else:
                return mu, var, np.sqrt(diag), nlpd
        else:
            if variance == 'diagonal':
                return mu, diag, np.sqrt(diag)
            else:
                return mu, var, np.sqrt(diag)


class CompressedMixtureOfLinearGaussians:
    # This class compresses the above mixture
    # for speed at prediction/deployment time

    def __init__(self, mixture, size=1000):
        self.mixture = mixture

        self.input_transform = self.mixture.input_transform
        self.target_transform = self.mixture.target_transform

        weights = self.mixture.gating.posterior.mean()
        idx = weights.argsort()[-size:][::-1]

        self.gating = {'weights': weights[idx]}

        _basis_mus = np.vstack([b.posterior_predictive_gaussian()[0]
                                    for b in self.mixture.basis])
        _basis_lmbdas = np.stack([b.posterior_predictive_gaussian()[1]
                                  for b in self.mixture.basis], axis=0)
        _basis_logdet_lmbdas = np.linalg.slogdet(_basis_lmbdas)[1]

        self.basis = {'mus': _basis_mus[idx, ...],
                      'lmbdas': _basis_lmbdas[idx, ...],
                      'logdet_lmbdas': _basis_logdet_lmbdas[idx, ...]}

        _models_mus = np.stack([m.posterior.matnorm.M for m in self.mixture.models], axis=0)
        self.models = {'Ms': _models_mus[idx, ...]}

    def log_basis_predictive(self, x):
        from mimo.util.stats import multivariate_gaussian_loglik as mvn_logpdf
        return mvn_logpdf(x, self.basis['mus'],
                          self.basis['lmbdas'],
                          self.basis['logdet_lmbdas'])

    def predictive_gating(self, x):
        log_basis_predictive = self.log_basis_predictive(x)
        effective_weights = self.gating['weights'] * np.exp(log_basis_predictive) + eps
        effective_weights = effective_weights / np.sum(effective_weights)
        return effective_weights

    def predictive_output(self, x):
        x = np.hstack((x, 1.))  # assumes affine input
        return np.einsum('nkh,h->nk', self.models['Ms'], x)

    def prediction(self, x):
        from mimo.util.data import transform, inverse_transform_mean

        x = np.squeeze(transform(np.atleast_2d(x), self.input_transform))

        weights = self.predictive_gating(x)
        mus = self.predictive_output(x)

        output = np.einsum('nk,n->k', mus, weights)
        return inverse_transform_mean(output, trans=self.target_transform)
