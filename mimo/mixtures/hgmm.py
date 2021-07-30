import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from mimo.utils.data import one_hot
from mimo.utils.stats import sample_discrete_from_log
from mimo.utils.data import batches

from tqdm import tqdm


class MixtureOfMixtureOfGaussians:

    def __init__(self, gating, components):

        self.gating = gating
        self.components = components
        self.resp = None

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
    def dim(self):
        return self.components.dim

    def used_labels(self, obs):
        raise NotImplementedError

    def rvs(self, size=1):
        labels = self.gating.rvs(size)
        counts = np.bincount(labels, minlength=self.size)

        obs = np.zeros((size, self.dim))
        for idx, (c, count) in enumerate(zip(self.components.dists, counts)):
            obs[labels == idx, ...], _ = c.rvs(count)

        perm = np.random.permutation(size)
        obs, labels = obs[perm], labels[perm]
        return obs, labels

    def log_likelihood(self, obs):
        log_lik = self.log_complete_likelihood(obs)
        return logsumexp(log_lik, axis=0)

    # Expectation-Maximization
    def log_complete_likelihood(self, obs):
        component_loglik = np.zeros((self.size, len(obs)))
        for m in range(self.size):
            component_loglik[m, :] = self.components[m].log_likelihood(obs)
        gating_loglik = self.gating.log_likelihood(np.arange(self.size))
        return component_loglik + np.expand_dims(gating_loglik, axis=1)

    def responsibilities(self, obs):
        log_lik = self.log_complete_likelihood(obs)
        resp = np.exp(log_lik - logsumexp(log_lik, axis=0, keepdims=True))
        return resp

    def max_likelihood(self, obs, maxiter=1, maxsubiter=10,
                       progressbar=True, processid=0):

        if self.resp is None:
            self.resp = npr.rand(self.size, len(obs))
            self.resp /= np.sum(self.resp, axis=0)

        log_lik = []
        with tqdm(total=maxiter, desc=f'EM #{processid + 1}',
                  position=processid, disable=not progressbar) as pbar:

            for _ in range(maxiter):
                # Maximization step
                for m in range(self.size):
                    self.components[m].max_likelihood(obs, self.resp[m, :],
                                                      maxiter=maxsubiter,
                                                      progressbar=False)
                self.gating.max_likelihood(None, self.resp)

                # Expectation step
                self.resp = self.responsibilities(obs)

                log_lik.append(np.sum(self.log_likelihood(obs)))
                pbar.update(1)

        return log_lik

    def plot(self, obs, labels=None,
             color=None, legend=False, alpha=None):

        if labels is None:
            resp = self.responsibilities(obs)
            labels = np.argmax(resp, axis=0)

        import matplotlib.pyplot as plt

        label_colors = ['red', 'blue',
                        'magenta', 'green',
                        'black']

        # plot data scatter
        plt.figure()
        for m in range(self.size):
            # plot parameters
            for k in range(self.components[m].size):
                self.components[m].components.dists[k].plot(color=label_colors[m])

            # plot data
            idx = np.where(labels == m)[0]
            plt.scatter(obs[idx, 0], obs[idx, 1], color=label_colors[m], marker='+')

        plt.show()


class BayesianMixtureOfGaussiansWithHierarchicalPrior:

    def __init__(self, gating, components):

        self.gating = gating
        self.components = components

        self.resp = None

        from mimo.mixtures import MixtureOfGaussians
        self.likelihood = MixtureOfGaussians(gating=self.gating.likelihood,
                                             components=self.components.likelihood)

    @property
    def size(self):
        return self.gating.dim

    @property
    def dim(self):
        return self.components.dim

    def used_labels(self, obs):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, obs, maxiter=1, maxsubiter=1,
                 progressbar=True, processid=0):

        with tqdm(total=maxiter, desc=f'Gibbs #{processid + 1}',
                  position=processid, disable=not progressbar) as pbar:

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
        gating_loglik = self.gating.expected_log_likelihood()
        return component_loglik + np.expand_dims(gating_loglik, axis=1)

    def expected_responsibilities(self, obs):
        log_lik = self.expected_log_complete_likelihood(obs)
        resp = np.exp(log_lik - logsumexp(log_lik, axis=0, keepdims=True))
        return resp

    def meanfield_coordinate_descent(self, obs=None, weights=None,
                                     maxiter=250, maxsubiter=10, tol=1e-8,
                                     progressbar=True, processid=0):

        if self.resp is None:
            self.resp = np.random.rand(self.size, len(obs))
            self.resp /= np.sum(self.resp, axis=0)

        vlb = []
        with tqdm(total=maxiter, desc=f'VI #{processid + 1}',
                  position=processid, disable=not progressbar) as pbar:

            for i in range(maxiter):
                self.resp = self.resp if weights is None else self.resp * weights

                self.meanfield_update_parameters(obs, self.resp, maxsubiter)
                self.resp = self.expected_responsibilities(obs)

                vlb.append(self.variational_lowerbound(obs, self.resp))

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

    def variational_lowerbound_obs(self, obs, resp):
        return np.sum(resp * self.components.expected_log_likelihood(obs))

    def variational_lowerbound_labels(self, resp):
        vlb = 0.
        vlb += np.einsum('kn,k->', resp, self.gating.expected_log_likelihood())
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

    def __init__(self, gating, components):

        self.gating = gating
        self.components = components
        self.resp = None

        from mimo.mixtures import MixtureOfGaussians

        gating_likelihood = self.gating.likelihood
        components_likelihood = []
        for m in range(self.size):
            _local_gating_likelihood = self.components[m].gating.likelihood
            _local_components_likelihood = self.components[m].components.likelihood
            _local_model_likelihood = MixtureOfGaussians(_local_gating_likelihood,
                                                         _local_components_likelihood)
            components_likelihood.append(_local_model_likelihood)

        self.likelihood = MixtureOfMixtureOfGaussians(gating=gating_likelihood,
                                                      components=components_likelihood)

    @property
    def size(self):
        return self.gating.dim

    @property
    def dim(self):
        return self.components.dim

    def used_labels(self, obs):
        raise NotImplementedError

    # Gibbs sampling
    def resample(self, obs,
                 maxiter=1, maxsubiter=50, maxsubsubiter=1,
                 progressbar=True, processid=0):

        with tqdm(total=maxiter, desc=f'Gibbs #{processid + 1}',
                  position=processid, disable=not progressbar) as pbar:

            for _ in range(maxiter):
                _, labels = self.resample_labels(obs)
                self.resample_gating(labels)
                self.resample_components(obs, labels,
                                         maxsubiter, maxsubsubiter)

                pbar.update(1)

    def resample_labels(self, obs):
        log_prob = self.likelihood.log_complete_likelihood(obs)
        labels = sample_discrete_from_log(log_prob, axis=0)
        return log_prob, labels

    def resample_gating(self, labels):
        self.gating.resample(labels)

    def resample_components(self, obs, labels, maxsubiter, maxsubsubiter):
        for m in range(self.size):
            idx = np.where(labels == m)[0]
            self.components[m].resample(obs=obs[idx],
                                        maxiter=maxsubiter,
                                        maxsubiter=maxsubsubiter,
                                        progressbar=False)

    def expected_log_complete_likelihood(self, obs):
        component_loglik = np.zeros((self.size, len(obs)))
        for m in range(self.size):
            component_loglik[m, :] = self.components[m].expected_log_likelihood(obs)
        gating_loglik = self.gating.expected_log_likelihood()
        return component_loglik + np.expand_dims(gating_loglik, axis=1)

    def expected_responsibilities(self, obs):
        log_prob = self.expected_log_complete_likelihood(obs)
        resp = np.exp(log_prob - logsumexp(log_prob, axis=0, keepdims=True))
        return resp

    def meanfield_coordinate_descent(self, obs,
                                     maxiter=250, maxsubiter=5, maxsubsubiter=5,
                                     tol=1e-8, progressbar=True, processid=0):

        if self.resp is None:
            self.resp = np.random.rand(self.size, len(obs))
            self.resp /= np.sum(self.resp, axis=0)

        vlb = []
        with tqdm(total=maxiter, desc=f'VI #{processid + 1}',
                  position=processid, disable=not progressbar) as pbar:

            for i in range(maxiter):
                self.meanfield_update_parameters(obs, self.resp,
                                                 maxsubiter, maxsubsubiter)
                self.resp = self.expected_responsibilities(obs)

                # vlb.append(self.variational_lowerbound(obs, upper_resp, lower_resp))
                #
                # if len(vlb) > 1:
                #     if abs(vlb[-1] - vlb[-2]) < tol:
                #         return vlb

                pbar.update(1)

        return vlb

    def meanfield_update_parameters(self, obs, resp, maxsubiter, maxsubsubiter):
        self.meanfield_update_components(obs, resp, maxsubiter, maxsubsubiter)
        self.meanfield_update_gating(resp)

    def meanfield_update_gating(self, resp):
        self.gating.meanfield_update(None, resp)

    def meanfield_update_components(self, obs, resp, maxsubiter, maxsubsubiter):
        for m in range(self.size):
            self.components[m].meanfield_coordinate_descent(obs=obs, weights=resp[m, :],
                                                            maxiter=maxsubiter,
                                                            maxsubiter=maxsubsubiter,
                                                            progressbar=False)

    def variational_lowerbound_upper_labels(self, resp):
        raise NotImplementedError

    def variational_lowerbound_lower_labels(self, resp):
        raise NotImplementedError

    def variational_lowerbound_obs(self, obs, resp):
        raise NotImplementedError

    # def variational_lowerbound(self, obs, upper_resp, lower_resp):
    #     vlb = 0.
    #     vlb += self.upper_gating.variational_lowerbound()
    #     vlb += self.lower_gating.variational_lowerbound()
    #     vlb += np.sum(self.components.variational_lowerbound())
    #     vlb += self.variational_lowerbound_upper_labels(upper_resp)
    #     vlb += self.variational_lowerbound_lower_labels(lower_resp)
    #     vlb += self.variational_lowerbound_obs(obs, upper_resp, lower_resp)
    #     return vlb

    def plot(self, obs, labels=None,
             color=None, legend=False, alpha=None):

        if labels is None:
            resp = self.expected_responsibilities(obs)
            labels = np.argmax(resp, axis=0)

        import matplotlib.pyplot as plt
        from matplotlib import cm

        # get colors
        cmap = cm.get_cmap('RdBu')
        if color is None:
            label_colors = dict((idx, cmap(v)) for idx, v in
                                enumerate(np.linspace(0, 1, self.size, endpoint=True)))
        else:
            label_colors = dict((idx, color) for idx in range(self.size))

        # plot data scatter
        plt.figure()
        for m in range(self.size):
            # plot parameters
            for k in range(self.components[m].size):
                self.components[m].components.likelihood.dists[k].plot(color=label_colors[m])

            # plot data
            idx = np.where(labels == m)[0]
            plt.scatter(obs[idx, 0], obs[idx, 1], color=label_colors[m], marker='+')

        plt.show()
