import numpy as np
import numpy.random as npr

from scipy import special as special
from scipy.special import logsumexp

from mimo.distributions.bayesian import CategoricalWithDirichlet
from mimo.distributions.bayesian import CategoricalWithStickBreaking

from mimo.utils.data import one_hot
from mimo.utils.stats import sample_discrete_from_log
from mimo.utils.data import batches

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

eps = np.finfo(np.float64).tiny


class MixtureOfLinearGaussians:

    def __init__(self, gating, basis, models):
        assert basis.size == gating.dim
        assert models.size == gating.dim

        self.gating = gating
        self.basis = basis  # input density
        self.models = models  # output density

    @property
    def params(self):
        raise NotImplementedError

    @property
    def nb_params(self):
        raise NotImplementedError

    @property
    def size(self):
        return self.gating.dim

    @property
    def input_dim(self):
        return self.models.input_dim

    @property
    def output_dim(self):
        return self.models.output_dim

    def used_labels(self, x, y):
        resp, _ = self.responsibilities(x, y)
        labels = np.argmax(resp, axis=0)
        label_usages = np.bincount(labels, minlength=self.size)
        used_labels = np.where(label_usages > 0)[0]
        return used_labels

    def rvs(self, size=1):
        z = self.gating.likelihood.rvs(size)
        counts = np.bincount(z, minlength=self.size)

        x = np.empty((size, self.input_dim))
        y = np.empty((size, self.output_dim))
        for idx, (b, m, count) in enumerate(zip(self.basis, self.models, counts)):
            x[z == idx, ...] = b.rvs(count)
            y[z == idx, ...] = m.rvs(x[z == idx, ...])

        perm = npr.permutation(size)
        x, y, z = x[perm], y[perm], z[perm]

        return x, y, z

    def log_likelihood(self, x, y):
        log_lik = self.log_complete_likelihood(x, y)
        return logsumexp(log_lik, axis=0)

    # Expectation-Maximization
    def log_complete_likelihood(self, x, y):
        basis_loglik = self.basis.log_likelihood(x)
        models_loglik = self.models.log_likelihood(x, y)
        gating_loglik = self.gating.log_likelihood(np.arange(self.size))
        return basis_loglik + models_loglik + np.expand_dims(gating_loglik, axis=1)

    def responsibilities(self, x, y):
        log_lik = self.log_complete_likelihood(x, y)
        resp = np.exp(log_lik - logsumexp(log_lik, axis=0, keepdims=True))
        return resp

    def max_likelihood(self, x, y, randomize=True,
                       maxiter=250, progressbar=True, processid=0):
        raise NotImplementedError


class BayesianMixtureOfLinearGaussians:

    def __init__(self, gating, basis, models, scale=False):
        assert basis.size == gating.dim
        assert models.size == gating.dim

        self.gating = gating
        self.basis = basis  # input density
        self.models = models  # output density

        self.likelihood = MixtureOfLinearGaussians(gating=self.gating.likelihood,
                                                   basis=self.basis.likelihood,
                                                   models=self.models.likelihood)

        self.scale = scale
        self.input_transform = StandardScaler()
        self.output_transform = StandardScaler()

    @property
    def size(self):
        return self.likelihood.size

    @property
    def input_dim(self):
        return self.likelihood.input_dim

    @property
    def output_dim(self):
        return self.likelihood.output_dim

    def used_labels(self, x, y):
        if self.scale:
            xx = self.input_transform.transform(x)
            yy = self.output_transform.transform(y)
        else:
            xx, yy = x, y

        resp = self.expected_responsibilities(xx, yy)
        z = np.argmax(resp, axis=0)
        label_usages = np.bincount(z, minlength=self.size)
        used_labels = np.where(label_usages > 0)[0]
        return used_labels

    def init_transform(self, x, y):
        self.scale = True
        self.input_transform.fit(x)
        self.output_transform.fit(y)

    def max_aposteriori(self, x, y, randomize=True, maxiter=250,
                        progressbar=True, processid=0):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, x, y, init_labels='prior',
                 maxiter=1, progressbar=True, processid=0):

        if self.scale:
            xx = self.input_transform.transform(x)
            yy = self.output_transform.transform(y)
        else:
            xx, yy = x, y

        if init_labels == 'random':
            z = npr.choice(self.size, size=(len(xx)))
        elif init_labels == 'posterior':
            _, z = self.resample_labels(xx, yy)
        elif init_labels == 'prior':
            z = self.gating.likelihood.rvs(len(xx))

        with tqdm(total=maxiter, desc=f'Gibbs #{processid + 1}',
                  position=processid, disable=not progressbar) as pbar:

            for _ in range(maxiter):
                self.resample_basis(xx, z)
                self.resample_models(xx, yy, z)
                self.resample_gating(z)
                _, z = self.resample_labels(xx, yy)

                pbar.update(1)

    def resample_labels(self, x, y):
        log_prob = self.likelihood.log_complete_likelihood(x, y)
        labels = sample_discrete_from_log(log_prob, axis=0)
        return log_prob, labels

    def resample_gating(self, z):
        self.gating.resample(z)

    def resample_basis(self, x, z):
        weights = one_hot(z, K=self.size)
        self.basis.resample(x, weights)

    def resample_models(self, x, y, z):
        weights = one_hot(z, K=self.size)
        self.models.resample(x, y, weights)

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

    def meanfield_coordinate_descent(self, x, y, randomize=True,
                                     maxiter=250, tol=1e-8,
                                     progressbar=True, processid=0):

        if self.scale:
            xx = self.input_transform.transform(x)
            yy = self.output_transform.transform(y)
        else:
            xx, yy = x, y

        if randomize:
            resp = npr.rand(self.size, len(xx))
            resp /= np.sum(resp, axis=0)
        else:
            resp = self.expected_responsibilities(xx, yy)

        vlb = []
        with tqdm(total=maxiter, desc=f'VI #{processid + 1}',
                  position=processid, disable=not progressbar) as pbar:

            for i in range(maxiter):
                self.meanfield_update_parameters(xx, yy, resp)
                resp = self.expected_responsibilities(xx, yy)

                vlb.append(self.variational_lowerbound(xx, yy, resp))

                if len(vlb) > 1:
                    if abs(vlb[-1] - vlb[-2]) < tol:
                        return vlb

                pbar.update(1)

        return vlb

    def meanfield_update_parameters(self, x, y, resp):
        self.meanfield_update_basis(x, resp)
        self.meanfield_update_models(x, y, resp)
        self.meanfield_update_gating(resp)

    def meanfield_update_gating(self, resp):
        self.gating.meanfield_update(None, resp)

    def meanfield_update_basis(self, x, resp):
        self.basis.meanfield_update(x, resp)

    def meanfield_update_models(self, x, y, resp):
        self.models.meanfield_update(x, y, resp)

    # SVI
    def meanfield_stochastic_descent(self, x, y, randomize=True,
                                     maxiter=500, stepsize=1e-3,
                                     batchsize=128, progressbar=True,
                                     procces_id=0):

        if self.scale:
            xx = self.input_transform.transform(x)
            yy = self.output_transform.transform(y)
        else:
            xx, yy = x, y

        vlb = []
        with tqdm(total=maxiter, desc=f'SVI #{procces_id + 1}',
                  position=procces_id, disable=not progressbar) as pbar:

            scale = batchsize / float(len(xx))
            for i in range(maxiter):
                for batch in batches(batchsize, len(xx)):
                    if i == 0 and randomize is True:
                        resp = npr.rand(self.size, len(xx[batch, :]))
                        resp /= np.sum(resp, axis=0)
                    else:
                        resp = self.expected_responsibilities(xx[batch, :], yy[batch, :])

                    self.meanfield_sgdstep_parameters(xx[batch, :], yy[batch, :],
                                                      resp, scale, stepsize)

                resp = self.expected_responsibilities(xx, yy)
                vlb.append(self.variational_lowerbound(xx, yy, resp))

                pbar.update(1)

        return vlb

    def meanfield_sgdstep_parameters(self, x, y, resp, scale, stepsize):
        self.meanfield_sgdstep_basis(x, resp, scale, stepsize)
        self.meanfield_sgdstep_models(x, y, resp, scale, stepsize)
        self.meanfield_sgdstep_gating(resp, scale, stepsize)

    def meanfield_sgdstep_gating(self, resp, scale, stepsize):
        self.gating.meanfield_sgdstep(None, resp, scale, stepsize)

    def meanfield_sgdstep_basis(self, x, resp, scale, stepsize):
        self.basis.meanfield_sgdstep(x, resp, scale, stepsize)

    def meanfield_sgdstep_models(self, x, y, resp, scale, stepsize):
        self.models.meanfield_sgdstep(x, y, resp, scale, stepsize)

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
        vlb += self.variational_lowerbound_data(x, y, resp)
        vlb += self.variational_lowerbound_labels(resp)
        return vlb

    def meanfield_predictive_activation(self, x, dist='gaussian'):
        # for plotting basis functions
        x = np.reshape(x, (-1, self.input_dim))
        xx = self.input_transform.transform(x) if self.scale else x

        log_basis = self.basis.log_posterior_predictive_gaussian(xx)\
            if dist == 'gaussian' else self.basis.log_posterior_predictive_studentt(xx)

        log_gating = np.log(self.gating.posterior.mean())
        log_activations = np.expand_dims(log_gating, axis=1) + log_basis
        activations = np.exp(log_activations - logsumexp(log_activations, axis=0, keepdims=True))

        return activations

    def meanfield_predictive_weights(self, x, dist='gaussian'):
        log_posterior_predictive = self.basis.log_posterior_predictive_gaussian(x)\
            if dist == 'gaussian' else self.basis.log_posterior_predictive_studentt(x)

        log_gating = np.log(self.gating.posterior.mean())
        log_weight = np.expand_dims(log_gating, axis=1) + log_posterior_predictive
        weights = np.exp(log_weight - logsumexp(log_weight, axis=0, keepdims=True))

        return weights

    def meanfield_predictive_moments(self, x, dist='gaussian'):
        mus, covars = np.zeros((self.size, len(x), self.output_dim)),\
                     np.zeros((self.size, len(x), self.output_dim, self.output_dim))

        if dist == 'gaussian':
            mus, lmbdas = self.models.posterior_predictive_gaussian(x)
            covars = np.linalg.inv(lmbdas)
        else:
            mus, lmbdas, dfs = self.models.posterior_predictive_studentt(x)
            covars = np.einsum('ndl,n->ndl', np.linalg.inv(lmbdas), dfs / (dfs - 2))

        return mus, covars

    def meanfiled_log_predictive_likelihood(self, x, y, dist='gaussian'):
        log_pl = self.models.log_posterior_predictive_gaussian(x, y) if dist == 'gaussian'\
            else self.models.log_posterior_predictive_studentt(x, y)
        return log_pl

    @staticmethod
    def mixture_moments(mus, covars, weights):
        # Mean of a mixture = sum of weighted means
        mu = np.einsum('knd,kn->nd', mus, weights)
        # Variance of a mixture = sum of weighted variances + ...
        # ... + sum of weighted squared means - squared sum of weighted means
        covar = np.einsum('kndl,kn->ndl', covars + np.einsum('knd,knl->kndl', mus, mus), weights)\
                - np.einsum('nd,nl->ndl', mu, mu)
        return mu, covar

    def meanfield_prediction(self, x, y=None,
                             prediction='average',
                             dist='gaussian',
                             incremental=False,
                             variance='diagonal'):

        x = np.reshape(x, (-1, self.input_dim))

        compute_nlpd = False
        if y is not None:
            y = np.reshape(y, (-1, self.output_dim))
            compute_nlpd = True

        if self.scale:
            xx = self.input_transform.transform(x)
            yy = self.output_transform.transform(y) if y is not None else None
        else:
            xx, yy = x, y if y is not None else None

        weights = self.meanfield_predictive_weights(xx, dist)
        mus, covars = self.meanfield_predictive_moments(xx, dist)

        if prediction == 'mode':
            k = np.argmax(weights, axis=0)
            idx = (k, range(len(k)), ...)
            mu, covar = mus[idx], covars[idx]
        elif prediction == 'average':
            mu, covar = self.mixture_moments(mus, covars, weights)
        else:
            raise NotImplementedError

        nlpd = None
        if compute_nlpd:
            log_pl = self.meanfiled_log_predictive_likelihood(xx, yy)
            log_weights = np.log(weights + eps)
            nlpd = - 1.0 * logsumexp(log_pl + log_weights, axis=0)

        if self.scale:
            mu = self.output_transform.inverse_transform(mu)
            mat = np.diag(np.sqrt(self.output_transform.var_))
            covar = np.einsum('kh,...hj,ji->...ki', mat, covar, mat.T)

        if incremental:
            mu += x[:, :self.output_dim]

        var = np.vstack(list(map(np.diag, covar)))

        if compute_nlpd:
            if variance == 'diagonal':
                return mu, var, np.sqrt(var), nlpd
            else:
                return mu, covar, np.sqrt(var), nlpd
        else:
            if variance == 'diagonal':
                return mu, var, np.sqrt(var)
            else:
                return mu, covar, np.sqrt(var)
