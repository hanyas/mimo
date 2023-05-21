import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from mimo.distributions.bayesian import CategoricalWithDirichlet
from mimo.distributions.bayesian import CategoricalWithStickBreaking

from mimo.utils.data import one_hot
from mimo.utils.stats import sample_discrete_from_log
from mimo.utils.data import batches

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


class MixtureOfMixtureOfLinearGaussians:

    def __init__(self, cluster_size, mixture_size,
                 input_dim, output_dim,
                 gating, components, scale=False):

        self.cluster_size = cluster_size
        self.mixture_size = mixture_size

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gating = gating
        self.components = components

        self.scale = scale
        self.input_transform = StandardScaler()
        self.output_transform = StandardScaler()

    @property
    def params(self):
        raise NotImplementedError

    @property
    def nb_params(self):
        raise NotImplementedError

    def init_transform(self, x, y):
        self.scale = True
        self.input_transform.fit(x)
        self.output_transform.fit(y)

    def used_labels(self, x, y):
        raise NotImplementedError

    def rvs(self, size=1):
        raise NotImplementedError

    def log_likelihood(self, x, y):
        log_lik = self.log_complete_likelihood(x, y)
        return logsumexp(log_lik, axis=0)

    # Expectation-Maximization
    def log_complete_likelihood(self, x, y):
        component_loglik = np.zeros((self.cluster_size, len(x)))
        for m in range(self.cluster_size):
            component_loglik[m] = self.components[m].log_likelihood(x, y)
        gating_loglik = self.gating.log_likelihood(np.arange(self.cluster_size))
        return component_loglik + np.expand_dims(gating_loglik, axis=1)

    def responsibilities(self, x, y):
        log_lik = self.log_complete_likelihood(x, y)
        resp = np.exp(log_lik - logsumexp(log_lik, axis=0, keepdims=True))
        return resp

    def max_likelihood(self, x, y, randomize=True,
                       maxiter=250, maxsubiter=5,
                       progress_bar=True, process_id=0):
        raise NotImplementedError


class BayesianMixtureOfLinearGaussiansWithTiedActivation:

    def __init__(self, size,
                 input_dim, output_dim,
                 gating, basis, models,
                 scale=False):

        self.size = size

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gating = gating
        self.basis = basis  # input density
        self.models = models  # output density

        from mimo.mixtures import MixtureOfLinearGaussians
        self.likelihood = MixtureOfLinearGaussians(self.size,
                                                   self.input_dim, self.output_dim,
                                                   gating=gating.likelihood,
                                                   basis=basis.likelihood,
                                                   models=models.likelihood)

    @property
    def params(self):
        raise NotImplementedError

    @property
    def nb_params(self):
        raise NotImplementedError

    def used_labels(self, x, y):
        raise NotImplementedError

    def rvs(self, size=1):
        raise NotImplementedError

    def max_aposteriori(self, x, y, randomize=True,
                        maxiter=250, progress_bar=True, process_id=0):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, x, y, maxiter=250, maxsubiter=5,
                 progress_bar=True, process_id=0):

        with tqdm(total=maxiter, desc=f'Init #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for _ in range(maxiter):
                _, z = self.resample_labels(x, y)
                self.resample_gating(z)
                self.resample_basis(x, z, maxsubiter)
                self.resample_models(x, y, z, maxsubiter)

                pbar.update(1)

    def resample_labels(self, x, y):
        log_prob = self.likelihood.log_complete_likelihood(x, y)
        labels = sample_discrete_from_log(log_prob, axis=0)
        return log_prob, labels

    def resample_gating(self, z):
        self.gating.resample(z)

    def resample_basis(self, x, z, maxsubiter):
        weights = one_hot(z, K=self.size)
        self.basis.resample(x, weights, maxsubiter)

    def resample_models(self, x, y, z, maxsubiter):
        weights = one_hot(z, K=self.size)
        self.models.resample(x, y, weights, maxsubiter)

    def expected_log_likelihood(self, x, y):
        log_lik = self.expected_log_complete_likelihood(x, y)
        return logsumexp(log_lik, axis=0)

    # Mean field
    def expected_log_complete_likelihood(self, x, y):
        basis_loglik = self.basis.expected_log_likelihood(x)
        models_loglik = self.models.expected_log_likelihood(x, y)

        gating_loglik = None
        if isinstance(self.gating, CategoricalWithDirichlet):
            gating_loglik = self.gating.expected_log_likelihood()
        elif isinstance(self.gating, CategoricalWithStickBreaking):
            log_stick, log_rest = self.gating.expected_log_likelihood()
            gating_loglik = log_stick + np.hstack((0, np.cumsum(log_rest)[:-1]))

        return basis_loglik + models_loglik + np.expand_dims(gating_loglik, axis=1)

    def expected_responsibilities(self, x, y):
        log_lik = self.expected_log_complete_likelihood(x, y)
        resp = np.exp(log_lik - logsumexp(log_lik, axis=0, keepdims=True))
        return resp

    # Mean field
    def meanfield_coordinate_descent(self, x, y, randomize=True,
                                     weights=None, maxiter=250,
                                     maxsubiter=5, tol=1e-16,
                                     progress_bar=True, process_id=0):

        if randomize:
            resp = npr.rand(self.size, len(x))
            resp /= np.sum(resp, axis=0)
        else:
            resp = self.expected_responsibilities(x, y)

        vlb = []
        with tqdm(total=maxiter, desc=f'VI #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for i in range(maxiter):
                resp = resp if weights is None else resp * weights

                self.meanfield_update_parameters(x, y, resp, maxsubiter)
                resp = self.expected_responsibilities(x, y)

                # vlb.append(self.variational_lowerbound(x, y, resp))

                if len(vlb) > 1:
                    if abs(vlb[-1] - vlb[-2]) < tol:
                        return vlb

                pbar.update(1)

        return vlb

    def meanfield_update_parameters(self, x, y, resp, maxsubiter):
        self.meanfield_update_basis(x, resp, maxsubiter)
        self.meanfield_update_models(x, y, resp, maxsubiter)
        self.meanfield_update_gating(resp)

    def meanfield_update_gating(self, resp):
        self.gating.meanfield_update(None, resp)

    def meanfield_update_basis(self, x, resp, maxsubiter):
        self.basis.meanfield_update(x, resp, maxsubiter)

    def meanfield_update_models(self, x, y, resp, maxsubiter):
        self.models.meanfield_update(x, y, resp, maxsubiter)

    def variational_lowerbound_data(self, x, y, resp):
        vlb = 0.
        vlb += np.sum(resp * self.basis.expected_log_likelihood(x))
        vlb += np.sum(resp * self.models.expected_log_likelihood(x, y))
        return vlb

    def variational_lowerbound_labels(self, resp):
        vlb = 0.
        if isinstance(self.gating, CategoricalWithDirichlet):
            vlb += np.sum(resp * np.expand_dims(self.gating.expected_log_likelihood(), axis=1))
        elif isinstance(self.gating, CategoricalWithStickBreaking):
            acc_resp = np.vstack((np.cumsum(resp[::-1, :], axis=0)[-2::-1, :],
                                  np.zeros((1, resp.shape[-1]))))
            E_log_stick, E_log_rest = self.gating.expected_log_likelihood()
            vlb += np.sum(resp * np.expand_dims(E_log_stick, axis=1)
                          + acc_resp * np.expand_dims(E_log_rest, axis=1))

        errs = np.seterr(invalid='ignore', divide='ignore')
        vlb -= np.nansum(resp * np.log(resp))
        np.seterr(**errs)
        return vlb

    def variational_lowerbound(self, x, y, resp):
        vlb = 0.
        vlb += self.gating.variational_lowerbound()
        vlb += np.sum(self.basis.variational_lowerbound())
        vlb += np.sum(self.models.variational_lowerbound())
        vlb += self.variational_lowerbound_labels(resp)
        vlb += self.variational_lowerbound_data(x, y, resp)
        return vlb


class BayesianMixtureOfMixtureOfLinearGaussians:

    def __init__(self, cluster_size, mixture_size,
                 input_dim, output_dim,
                 gating, components, scale=False):

        self.cluster_size = cluster_size
        self.mixture_size = mixture_size

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gating = gating
        self.components = components

        gating_likelihood = self.gating.likelihood
        components_likelihood = []
        for m in range(self.cluster_size):
            components_likelihood.append(self.components[m].likelihood)

        self.likelihood = MixtureOfMixtureOfLinearGaussians(self.cluster_size, self.mixture_size,
                                                            self.input_dim, self.output_dim,
                                                            gating=gating_likelihood,
                                                            components=components_likelihood)

        self.scale = scale
        self.input_transform = StandardScaler()
        self.output_transform = StandardScaler()

    def used_labels(self, x, y):
        if self.scale:
            xx = self.input_transform.transform(x)
            yy = self.output_transform.transform(y)
        else:
            xx, yy = x, y

        resp = self.expected_responsibilities(xx, yy)
        z = np.argmax(resp, axis=0)
        label_usages = np.bincount(z, minlength=self.cluster_size)
        used_labels = np.where(label_usages > 0)[0]
        return used_labels

    def init_transform(self, x, y):
        self.scale = True
        self.input_transform.fit(x)
        self.output_transform.fit(y)

    def max_aposteriori(self, x, y, randomize=True,
                        maxiter=250, maxsubiter=5,
                        progress_bar=True, process_id=0):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, x, y, init_labels='prior',
                 maxiter=250, maxsubiter=100, maxsubsubiter=5,
                 progress_bar=True, process_id=0):

        if self.scale:
            xx = self.input_transform.transform(x)
            yy = self.output_transform.transform(y)
        else:
            xx, yy = x, y

        if init_labels == 'random':
            z = npr.choice(self.cluster_size, size=(len(xx)))
        elif init_labels == 'posterior':
            _, z = self.resample_labels(xx, yy)
        elif init_labels == 'prior':
            z = self.gating.likelihood.rvs(len(xx))

        with tqdm(total=maxiter, desc=f'Init #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for _ in range(maxiter):
                self.resample_components(xx, yy, z, maxsubiter, maxsubsubiter)
                self.resample_gating(z)
                _, z = self.resample_labels(xx, yy)

                pbar.update(1)

    def resample_labels(self, x, y):
        log_prob = self.likelihood.log_complete_likelihood(x, y)
        labels = sample_discrete_from_log(log_prob, axis=0)
        return log_prob, labels

    def resample_gating(self, z):
        self.gating.resample(z)

    def resample_components(self, x, y, z, maxsubiter, maxsubsubiter):
        for m in range(self.cluster_size):
            idx = np.where(z == m)[0]
            self.components[m].resample(x=x[idx], y=y[idx],
                                        maxiter=maxsubiter,
                                        maxsubiter=maxsubsubiter,
                                        progress_bar=False)

    def expected_log_complete_likelihood(self, x, y):
        component_loglik = np.zeros((self.cluster_size, len(x)))
        for m in range(self.cluster_size):
            component_loglik[m, :] = self.components[m].expected_log_likelihood(x, y)

        gating_loglik = None
        if isinstance(self.gating, CategoricalWithDirichlet):
            gating_loglik = self.gating.expected_log_likelihood()
        elif isinstance(self.gating, CategoricalWithStickBreaking):
            log_stick, log_rest = self.gating.expected_log_likelihood()
            gating_loglik = log_stick + np.hstack((0, np.cumsum(log_rest)[:-1]))

        return component_loglik + np.expand_dims(gating_loglik, axis=1)

    def expected_responsibilities(self, x, y):
        log_lik = self.expected_log_complete_likelihood(x, y)
        resp = np.exp(log_lik - logsumexp(log_lik, axis=0, keepdims=True))
        return resp

    def meanfield_coordinate_descent(self, x, y, randomize=True,
                                     maxiter=250, maxsubiter=5, maxsubsubiter=5,
                                     tol=1e-16, progress_bar=True, process_id=0):

        if self.scale:
            xx = self.input_transform.transform(x)
            yy = self.output_transform.transform(y)
        else:
            xx, yy = x, y

        if randomize:
            resp = npr.rand(self.cluster_size, len(xx))
            resp /= np.sum(resp, axis=0)
        else:
            resp = self.expected_responsibilities(xx, yy)

        vlb = []
        with tqdm(total=maxiter, desc=f'VI #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for i in range(maxiter):
                randomize = randomize if i == 0 else False
                self.meanfield_update_parameters(xx, yy, resp, maxsubiter,
                                                 maxsubsubiter, randomize)
                resp = self.expected_responsibilities(xx, yy)

                pbar.update(1)

        return vlb

    def meanfield_update_parameters(self, x, y, resp, maxsubiter,
                                    maxsubsubiter, randomize):
        self.meanfield_update_components(x, y, resp, maxsubiter,
                                         maxsubsubiter, randomize)
        self.meanfield_update_gating(resp)

    def meanfield_update_gating(self, resp):
        self.gating.meanfield_update(None, resp)

    def meanfield_update_components(self, x, y, resp, maxsubiter,
                                    maxsubsubiter, randomize):

        for m in range(self.cluster_size):
            self.components[m].meanfield_coordinate_descent(x=x, y=y,
                                                            randomize=randomize,
                                                            weights=resp[m, :],
                                                            maxiter=maxsubiter,
                                                            maxsubiter=maxsubsubiter,
                                                            progress_bar=False)

    def variational_lowerbound_labels(self, resp):
        raise NotImplementedError

    def variational_lowerbound_data(self, x, y, resp):
        raise NotImplementedError

    def variational_lowerbound(self, x, y, upper_resp, lower_resp):
        raise NotImplementedError

    # for plotting basis functions
    def meanfield_predictive_activation(self, x):
        x = np.reshape(x, (-1, self.input_dim))
        xx = self.input_transform.transform(x) if self.scale else x

        log_basis = np.zeros((self.cluster_size, self.mixture_size, len(xx)))
        for m in range(self.cluster_size):
            log_basis[m] = np.expand_dims(np.log(self.components[m].gating.posterior.mean()), axis=1)\
                           + self.components[m].basis.log_posterior_predictive_gaussian(xx)

        log_gating = np.log(self.gating.posterior.mean())

        log_activations = log_basis + np.expand_dims(log_gating, axis=(1, 2))
        activations = np.exp(log_activations - logsumexp(log_activations, axis=(0, 1), keepdims=True))
        return activations

    def meanfield_predictive_weights(self, x):
        log_cluster_gating = np.log(self.gating.posterior.mean())

        log_weights = np.zeros((self.cluster_size, self.mixture_size, len(x)))
        for m in range(self.cluster_size):
            log_weights[m] = np.expand_dims(np.log(self.components[m].gating.posterior.mean()), axis=1) \
                             + self.components[m].basis.log_posterior_predictive_gaussian(x) + log_cluster_gating[m]

        weights = np.exp(log_weights - logsumexp(log_weights, axis=(0, 1), keepdims=True))
        return weights

    def meanfield_predictive_moments(self, x):
        mus = np.zeros((self.cluster_size, self.mixture_size, len(x), self.output_dim))
        lmbdas = np.zeros((self.cluster_size, self.mixture_size, len(x), self.output_dim, self.output_dim))
        for m in range(self.cluster_size):
            mus[m], lmbdas[m] = self.components[m].models.posterior_predictive_gaussian(x)

        sigmas = np.linalg.inv(lmbdas)
        return mus, sigmas

    @staticmethod
    def mixture_moments(mus, covars, weights):
        # Mean of a mixture = sum of weighted means
        mean = np.einsum('mknd,mkn->nd', mus, weights)
        # Variance of a mixture = sum of weighted variances + ...
        # ... + sum of weighted squared means - squared sum of weighted means
        covar = np.einsum('mkndl,mkn->ndl', covars + np.einsum('mknd,mknl->mkndl', mus, mus), weights)\
                - np.einsum('nd,nl->ndl', mean, mean)
        return mean, covar

    def meanfield_prediction(self, x,
                             prediction='average',
                             incremental=False,
                             variance='diagonal'):

        x = np.reshape(x, (-1, self.input_dim))
        xx = self.input_transform.transform(x) if self.scale else x

        weights = self.meanfield_predictive_weights(xx)
        mus, sigmas = self.meanfield_predictive_moments(xx)

        if prediction == 'mode':
            _weights = np.reshape(weights, (-1, len(xx)))
            _mus = np.reshape(mus, (-1, len(xx), self.output_dim))
            _sigmas = np.reshape(sigmas, (-1, len(xx), self.output_dim, self.output_dim))

            mk = np.argmax(_weights, axis=0)
            idx = (mk, range(len(mk)), ...)
            mean, covar = _mus[idx], _sigmas[idx]
        elif prediction == 'average':
            mean, covar = self.mixture_moments(mus, sigmas, weights)
        else:
            raise NotImplementedError

        if self.scale:
            mean = self.output_transform.inverse_transform(mean)
            mat = np.diag(np.sqrt(self.output_transform.var_))
            covar = np.einsum('kh,...hj,ji->...ki', mat, covar, mat.T)

        if incremental:
            mean += x[:, :self.output_dim]

        var = np.vstack(list(map(np.diag, covar)))

        if variance == 'diagonal':
            return mean, var, np.sqrt(var)
        else:
            return mean, covar, np.sqrt(var)
