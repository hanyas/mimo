import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from mimo.distributions.bayesian import CategoricalWithDirichlet
from mimo.distributions.bayesian import CategoricalWithStickBreaking

from mimo.utils.data import one_hot
from mimo.utils.stats import sample_discrete_from_log
from mimo.utils.data import batches

from tqdm import tqdm


class MixtureOfMixtureOfGaussians:

    def __init__(self, cluster_size, mixture_size,
                 dim, gating, components):

        self.cluster_size = cluster_size
        self.mixture_size = mixture_size
        self.dim = dim

        self.gating = gating
        self.components = components

    @property
    def params(self):
        raise NotImplementedError

    @property
    def nb_params(self):
        raise NotImplementedError

    def used_labels(self, obs):
        raise NotImplementedError

    def rvs(self, size=1):
        raise NotImplementedError

    def log_likelihood(self, obs):
        log_lik = self.log_complete_likelihood(obs)
        return logsumexp(log_lik, axis=0)

    # Expectation-Maximization
    def log_complete_likelihood(self, obs):
        component_loglik = np.zeros((self.cluster_size, len(obs)))
        for m in range(self.cluster_size):
            component_loglik[m, :] = self.components[m].log_likelihood(obs)
        gating_loglik = self.gating.log_likelihood(np.arange(self.cluster_size))
        return component_loglik + np.expand_dims(gating_loglik, axis=1)

    def responsibilities(self, obs):
        log_lik = self.log_complete_likelihood(obs)
        resp = np.exp(log_lik - logsumexp(log_lik, axis=0, keepdims=True))
        return resp

    def max_likelihood(self, obs, randomize=True,
                       maxiter=250, maxsubiter=5,
                       progress_bar=True, process_id=0):

        if randomize:
            resp = npr.rand(self.cluster_size, len(obs))
            resp /= np.sum(resp, axis=0)
        else:
            resp = self.responsibilities(obs)

        log_lik = []
        with tqdm(total=maxiter, desc=f'EM #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for i in range(maxiter):
                # Maximization step
                for m in range(self.cluster_size):
                    randomize = randomize if i == 0 else False
                    self.components[m].max_likelihood(obs, weights=resp[m, :],
                                                      randomize=randomize,
                                                      maxiter=maxsubiter,
                                                      progress_bar=False)
                self.gating.max_likelihood(None, resp)

                # Expectation step
                resp = self.responsibilities(obs)

                log_lik.append(np.sum(self.log_likelihood(obs)))
                pbar.update(1)

        return log_lik

    def plot(self, obs, labels=None):

        if labels is None:
            resp = self.responsibilities(obs)
            labels = np.argmax(resp, axis=0)

        import matplotlib.pyplot as plt

        label_colors = ['red', 'blue',
                        'magenta', 'green',
                        'black', 'orange',
                        'purple', 'brown']

        # plot data scatter
        plt.figure()
        for m in range(self.cluster_size):
            # plot parameters
            for k in range(self.mixture_size):
                self.components[m].components.dists[k].plot(color=label_colors[m])

            # plot data
            idx = np.where(labels == m)[0]
            plt.scatter(obs[idx, 0], obs[idx, 1], alpha=0.1, color=label_colors[m], marker='+')

        plt.show()


class BayesianMixtureOfGaussiansWithHierarchicalPrior:

    def __init__(self, size, dim,
                 gating, components):

        self.size = size
        self.dim = dim

        self.gating = gating
        self.components = components

        from mimo.mixtures import MixtureOfGaussians
        self.likelihood = MixtureOfGaussians(gating=self.gating.likelihood,
                                             components=self.components.likelihood)

    def used_labels(self, obs):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, obs, maxiter=250, maxsubiter=5,
                 progress_bar=True, process_id=0):

        with tqdm(total=maxiter, desc=f'Init #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for _ in range(maxiter):
                _, labels = self.resample_labels(obs)
                self.resample_gating(labels)
                self.resample_components(obs, labels,
                                         maxsubiter)

                pbar.update(1)

    def resample_labels(self, obs):
        log_prob = self.likelihood.log_complete_likelihood(obs)
        labels = sample_discrete_from_log(log_prob, axis=0)
        return log_prob, labels

    def resample_gating(self, labels):
        self.gating.resample(labels)

    def resample_components(self, obs, labels, maxsubiter):
        weights = one_hot(labels, K=self.size)
        self.components.resample(obs, weights, maxsubiter)

    def expected_log_likelihood(self, obs):
        log_lik = self.expected_log_complete_likelihood(obs)
        return logsumexp(log_lik, axis=0)

    # Mean field
    def expected_log_complete_likelihood(self, obs):
        component_loglik = self.components.expected_log_likelihood(obs)

        gating_loglik = None
        if isinstance(self.gating, CategoricalWithDirichlet):
            gating_loglik = self.gating.expected_log_likelihood()
        elif isinstance(self.gating, CategoricalWithStickBreaking):
            log_stick, log_rest = self.gating.expected_log_likelihood()
            gating_loglik = log_stick + np.hstack((0, np.cumsum(log_rest)[:-1]))

        return component_loglik + np.expand_dims(gating_loglik, axis=1)

    def expected_responsibilities(self, obs):
        log_lik = self.expected_log_complete_likelihood(obs)
        resp = np.exp(log_lik - logsumexp(log_lik, axis=0, keepdims=True))
        return resp

    # Mean field
    def meanfield_coordinate_descent(self, obs, randomize=True,
                                     weights=None, maxiter=250,
                                     maxsubiter=5, tol=1e-8,
                                     progress_bar=True, process_id=0):

        if randomize:
            resp = npr.rand(self.size, len(obs))
            resp /= np.sum(resp, axis=0)
        else:
            resp = self.expected_responsibilities(obs)

        vlb = []
        with tqdm(total=maxiter, desc=f'VI #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for i in range(maxiter):
                resp = resp if weights is None else resp * weights

                self.meanfield_update_parameters(obs, resp, maxsubiter)
                resp = self.expected_responsibilities(obs)

                vlb.append(self.variational_lowerbound(obs, resp))

                if len(vlb) > 1:
                    if abs(vlb[-1] - vlb[-2]) < tol:
                        return vlb

                pbar.update(1)

        return vlb

    def meanfield_update_parameters(self, obs, resp, maxsubiter):
        self.meanfield_update_components(obs, resp, maxsubiter)
        self.meanfield_update_gating(resp)

    def meanfield_update_gating(self, resp):
        self.gating.meanfield_update(None, resp)

    def meanfield_update_components(self, obs, resp, maxsubiter):
        self.components.meanfield_update(obs, resp, maxsubiter)

    # SVI
    def meanfield_stochastic_descent(self, obs, randomize=True,
                                     weights=None, maxiter=250,
                                     maxsubiter=5, scale=1, step_size=1e-2,
                                     progress_bar=True, procces_id=0):

        if randomize is True:
            resp = npr.rand(self.size, len(obs))
            resp /= np.sum(resp, axis=0)
        else:
            resp = self.expected_responsibilities(obs)

        vlb = []
        with tqdm(total=maxiter, desc=f'SVI #{procces_id + 1}',
                  position=procces_id, disable=not progress_bar) as pbar:

            for i in range(maxiter):
                resp = resp if weights is None else resp * weights

                self.meanfield_sgdstep_parameters(obs, resp, maxsubiter,
                                                  scale, step_size)
                resp = self.expected_responsibilities(obs)

                pbar.update(1)

        return vlb

    def meanfield_sgdstep_parameters(self, obs, resp, maxsubiter, scale, step_size):
        self.meanfield_sgdstep_components(obs, resp, maxsubiter, scale, step_size)
        self.meanfield_sgdstep_gating(resp, scale, step_size)

    def meanfield_sgdstep_components(self, obs, resp, maxsubiter, scale, step_size):
        self.components.meanfield_sgdstep(obs, resp, maxsubiter, scale, step_size)

    def meanfield_sgdstep_gating(self, resp, scale, step_size):
        self.gating.meanfield_sgdstep(None, resp, scale, step_size)

    def variational_lowerbound_obs(self, obs, resp):
        return np.sum(resp * self.components.expected_log_likelihood(obs))

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

    def variational_lowerbound(self, obs, resp):
        vlb = 0.
        vlb += self.gating.variational_lowerbound()
        vlb += np.sum(self.components.variational_lowerbound())
        vlb += self.variational_lowerbound_labels(resp)
        vlb += self.variational_lowerbound_obs(obs, resp)
        return vlb

    def plot(self, obs, labels=None, color=None, legend=False, alpha=None):
        resp = self.expected_responsibilities(obs)
        labels = np.argmax(resp, axis=0)
        artists = self.likelihood.plot(obs, labels, color, legend, alpha)
        return artists


class BayesianMixtureOfMixtureOfGaussians:
    """
    class for a Bayesian mixture of mixture of Gaussians
    """

    def __init__(self, cluster_size, mixture_size,
                 dim, gating, components):

        self.cluster_size = cluster_size
        self.mixture_size = mixture_size
        self.dim = dim

        self.gating = gating
        self.components = components

        gating_likelihood = self.gating.likelihood
        components_likelihood = []
        for m in range(self.cluster_size):
            components_likelihood.append(self.components[m].likelihood)

        self.likelihood = MixtureOfMixtureOfGaussians(cluster_size, mixture_size, dim,
                                                      gating=gating_likelihood,
                                                      components=components_likelihood)

    def used_labels(self, obs):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, obs, init_labels='prior',
                 maxiter=250, maxsubiter=100, maxsubsubiter=5,
                 progress_bar=True, process_id=0):

        if init_labels == 'random':
            labels = npr.choice(self.cluster_size, size=(len(obs)))
        elif init_labels == 'prior':
            labels = self.gating.likelihood.rvs(len(obs))
        elif init_labels == 'posterior':
            _, labels = self.resample_labels(obs)

        with tqdm(total=maxiter, desc=f'Init #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for _ in range(maxiter):
                self.resample_components(obs, labels, maxsubiter, maxsubsubiter)
                self.resample_gating(labels)
                _, labels = self.resample_labels(obs)

                pbar.update(1)

    def resample_labels(self, obs):
        log_prob = self.likelihood.log_complete_likelihood(obs)
        labels = sample_discrete_from_log(log_prob, axis=0)
        return log_prob, labels

    def resample_gating(self, labels):
        self.gating.resample(labels)

    def resample_components(self, obs, labels, maxsubiter, maxsubsubiter):
        for m in range(self.cluster_size):
            idx = np.where(labels == m)[0]
            self.components[m].resample(obs=obs[idx],
                                        maxiter=maxsubiter,
                                        maxsubiter=maxsubsubiter,
                                        progress_bar=False)

    def expected_log_complete_likelihood(self, obs):
        component_loglik = np.zeros((self.cluster_size, len(obs)))
        for m in range(self.cluster_size):
            component_loglik[m, :] = self.components[m].expected_log_likelihood(obs)

        gating_loglik = None
        if isinstance(self.gating, CategoricalWithDirichlet):
            gating_loglik = self.gating.expected_log_likelihood()
        elif isinstance(self.gating, CategoricalWithStickBreaking):
            log_stick, log_rest = self.gating.expected_log_likelihood()
            gating_loglik = log_stick + np.hstack((0, np.cumsum(log_rest)[:-1]))

        return component_loglik + np.expand_dims(gating_loglik, axis=1)

    def expected_responsibilities(self, obs):
        log_prob = self.expected_log_complete_likelihood(obs)
        resp = np.exp(log_prob - logsumexp(log_prob, axis=0, keepdims=True))
        return resp

    def meanfield_coordinate_descent(self, obs, randomize=True,
                                     maxiter=250, maxsubiter=5, maxsubsubiter=5,
                                     tol=1e-8, progress_bar=True, process_id=0):

        if randomize:
            resp = npr.rand(self.cluster_size, len(obs))
            resp /= np.sum(resp, axis=0)
        else:
            resp = self.expected_responsibilities(obs)

        vlb = []
        with tqdm(total=maxiter, desc=f'VI #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for i in range(maxiter):
                randomize = randomize if i == 0 else False
                self.meanfield_update_parameters(obs, resp, maxsubiter,
                                                 maxsubsubiter, randomize)
                resp = self.expected_responsibilities(obs)

                # vlb.append(self.variational_lowerbound(obs, upper_resp, lower_resp))
                #
                # if len(vlb) > 1:
                #     if abs(vlb[-1] - vlb[-2]) < tol:
                #         return vlb

                pbar.update(1)

        return vlb

    def meanfield_update_parameters(self, obs, resp, maxsubiter,
                                    maxsubsubiter, randomize):

        self.meanfield_update_gating(resp)
        self.meanfield_update_components(obs, resp, maxsubiter,
                                         maxsubsubiter, randomize)

    def meanfield_update_gating(self, resp):
        self.gating.meanfield_update(None, resp)

    def meanfield_update_components(self, obs, resp, maxsubiter,
                                    maxsubsubiter, randomize):

        for m in range(self.cluster_size):
            self.components[m].meanfield_coordinate_descent(obs=obs,
                                                            randomize=randomize,
                                                            weights=resp[m, :],
                                                            maxiter=maxsubiter,
                                                            maxsubiter=maxsubsubiter,
                                                            progress_bar=False)

    # SVI
    def meanfield_stochastic_descent(self, obs, randomize=True,
                                     maxiter=250, maxsubiter=5, maxsubsubiter=5,
                                     step_size=1e-2, batch_size=128, progress_bar=True,
                                     procces_id=0):

        vlb = []
        with tqdm(total=maxiter, desc=f'SVI #{procces_id + 1}',
                  position=procces_id, disable=not progress_bar) as pbar:

            scale = batch_size / float(len(obs))
            for i in range(maxiter):
                randomize = randomize if i == 0 else False
                for batch in batches(batch_size, len(obs)):
                    if randomize is True:
                        resp = npr.rand(self.cluster_size, len(obs[batch, :]))
                        resp /= np.sum(resp, axis=0)
                    else:
                        resp = self.expected_responsibilities(obs[batch, :])

                    self.meanfield_sgdstep_parameters(obs[batch, :], resp,
                                                      maxsubiter, maxsubsubiter,
                                                      randomize, scale, step_size)

                pbar.update(1)

        return vlb

    def meanfield_sgdstep_parameters(self, obs, resp,
                                     maxsubiter, maxsubsubiter,
                                     randomize, scale, step_size):
        self.meanfield_sgdstep_components(obs, resp,
                                          maxsubiter, maxsubsubiter,
                                          randomize, scale, step_size)
        self.meanfield_sgdstep_gating(resp, scale, step_size)

    def meanfield_sgdstep_components(self, obs, resp,
                                     maxsubiter, maxsubsubiter,
                                     randomize, scale, step_size):

        for m in range(self.cluster_size):
            self.components[m].meanfield_stochastic_descent(obs=obs,
                                                            randomize=randomize,
                                                            weights=resp[m, :],
                                                            maxiter=maxsubiter,
                                                            maxsubiter=maxsubsubiter,
                                                            scale=scale, step_size=step_size,
                                                            progress_bar=False)

    def meanfield_sgdstep_gating(self, resp, scale, step_size):
        self.gating.meanfield_sgdstep(None, resp, scale, step_size)

    def variational_lowerbound_labels(self, resp):
        raise NotImplementedError

    def variational_lowerbound_obs(self, obs, resp):
        raise NotImplementedError

    # def variational_lowerbound(self, obs, resp):
    #     vlb = 0.
    #     vlb += self.gating.variational_lowerbound()
    #     vlb += np.sum(self.components.variational_lowerbound())
    #     vlb += self.variational_lowerbound_labels(resp)
    #     vlb += self.variational_lowerbound_obs(obs, resp)
    #     return vlb

    def plot(self, obs, labels=None):
        if labels is None:
            resp = self.expected_responsibilities(obs)
            labels = np.argmax(resp, axis=0)

        self.likelihood.plot(obs, labels)
