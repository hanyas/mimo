import numpy as np
import numpy.random as npr

from scipy.special import logsumexp

from mimo.distributions.bayesian import CategoricalWithDirichlet
from mimo.distributions.bayesian import CategoricalWithStickBreaking

from mimo.utils.data import one_hot
from mimo.utils.stats import sample_discrete_from_log
from mimo.utils.data import batches

from tqdm import tqdm


class MixtureOfGaussians:
    """
    class for mixtures of Gaussians.
    """

    def __init__(self, gating, components):
        assert components.size == gating.dim

        self.gating = gating
        self.components = components

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
        resp, _ = self.responsibilities(obs)
        labels = np.argmax(resp, axis=0)
        label_usages = np.bincount(labels, minlength=self.size)
        used_labels = np.where(label_usages > 0)[0]
        return used_labels

    def rvs(self, size=1):
        labels = self.gating.rvs(size)
        counts = np.bincount(labels, minlength=self.size)

        obs = np.zeros((size, self.dim))
        for idx, (c, count) in enumerate(zip(self.components.dists, counts)):
            obs[labels == idx, ...] = c.rvs(count)

        perm = npr.permutation(size)
        obs, labels = obs[perm], labels[perm]
        return obs, labels

    def log_likelihood(self, obs):
        log_lik = self.log_complete_likelihood(obs)
        return logsumexp(log_lik, axis=0)

    # Expectation-Maximization
    def log_complete_likelihood(self, obs):
        component_loglik = self.components.log_likelihood(obs)
        gating_loglik = self.gating.log_likelihood(np.arange(self.size))
        return component_loglik + np.expand_dims(gating_loglik, axis=1)

    def responsibilities(self, obs):
        log_lik = self.log_complete_likelihood(obs)
        resp = np.exp(log_lik - logsumexp(log_lik, axis=0, keepdims=True))
        return resp

    def max_likelihood(self, obs, randomize=True, weights=None,
                       maxiter=250, progress_bar=True, process_id=0):

        if randomize:
            resp = npr.rand(self.size, len(obs))
            resp /= np.sum(resp, axis=0)
        else:
            resp = self.responsibilities(obs)

        log_lik = []
        with tqdm(total=maxiter, desc=f'EM #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for _ in range(maxiter):
                resp = resp if weights is None else resp * weights

                # Maximization step
                self.components.max_likelihood(obs, resp)
                self.gating.max_likelihood(None, resp)

                # Expectation step
                resp = self.responsibilities(obs)

                log_lik.append(np.sum(self.log_likelihood(obs)))
                pbar.update(1)

        return log_lik

    def plot(self, obs, labels=None,
             color=None, legend=False, alpha=None):

        if labels is None:
            resp = self.responsibilities(obs)
            labels = np.argmax(resp, axis=0)

        import matplotlib.pyplot as plt
        from matplotlib import cm

        artists = []

        # get colors
        cmap = cm.get_cmap('RdBu')
        if color is None:
            label_colors = dict((idx, cmap(v)) for idx, v in
                                enumerate(np.linspace(0, 1, self.size, endpoint=True)))
        else:
            label_colors = dict((idx, color) for idx in range(self.size))

        # plot data scatter
        color_seq = [label_colors[l] for l in labels]
        artists.append(plt.scatter(obs[:, 0], obs[:, 1], c=color_seq, marker='+'))

        # plot parameters
        axis = plt.axis()
        for k, (c, w) in enumerate(zip(self.components.dists, self.gating.probs)):
            artists.extend(c.plot(color=label_colors[k], label='%d' % k,
                                  alpha=min(0.25, 1. - (1. - w) ** 2) / 0.25
                                  if alpha is None else alpha))
        plt.axis(axis)

        # add legend
        if legend and color is None:
            plt.legend([plt.Rectangle((0, 0), 1, 1, fc=c)
                        for i, c in label_colors.items() if i in self.used_labels],
                       [i for i in label_colors if i in self.used_labels], loc='best', ncol=2)
        plt.show()

        return artists


class BayesianMixtureOfGaussians:
    """
    class for a Bayesian mixtures of Gaussians.
    """

    def __init__(self, gating, components):

        self.gating = gating
        self.components = components

        self.likelihood = MixtureOfGaussians(gating=self.gating.likelihood,
                                             components=self.components.likelihood)

    @property
    def size(self):
        return self.likelihood.size

    @property
    def dim(self):
        return self.likelihood.dim

    def used_labels(self, obs):
        resp = self.expected_responsibilities(obs)
        labels = np.argmax(resp, axis=0)
        label_usages = np.bincount(labels, minlength=self.size)
        used_labels = np.where(label_usages > 0)[0]
        return used_labels

    # Expectation-Maximization
    def max_aposteriori(self, obs, randomize=True, maxiter=250,
                        progress_bar=True, process_id=0):

        if randomize:
            resp = npr.rand(self.size, len(obs))
            resp /= np.sum(resp, axis=0)
        else:
            resp = self.likelihood.responsibilities(obs)

        log_prob = []
        with tqdm(total=maxiter, desc=f'MAP #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for i in range(maxiter):
                # Maximization step
                self.components.max_aposteriori(obs, resp)
                self.gating.max_aposteriori(None, resp)

                # Expectation step
                resp = self.likelihood.responsibilities(obs)

                log_lik = np.sum(self.likelihood.log_likelihood(obs))
                log_prior = self.gating.prior.log_likelihood(self.gating.likelihood.params)\
                            + np.sum(self.components.prior.log_likelihood(self.components.likelihood.params))
                log_prob.append(log_lik + log_prior)

                pbar.update(1)

        return log_prob

    # Gibbs sampling
    def resample(self, obs, init_labels='prior',
                 maxiter=1, progress_bar=True, process_id=0):

        if init_labels == 'random':
            labels = npr.choice(self.size, size=(len(obs)))
        elif init_labels == 'prior':
            labels = self.gating.likelihood.rvs(len(obs))
        elif init_labels == 'posterior':
            _, labels = self.resample_labels(obs)

        with tqdm(total=maxiter, desc=f'Gibbs #{process_id + 1}',
                  position=process_id, disable=not progress_bar) as pbar:

            for _ in range(maxiter):
                self.resample_components(obs, labels)
                self.resample_gating(labels)
                _, labels = self.resample_labels(obs)

                pbar.update(1)

    def resample_labels(self, obs):
        log_prob = self.likelihood.log_complete_likelihood(obs)
        labels = sample_discrete_from_log(log_prob, axis=0)
        return log_prob, labels

    def resample_gating(self, labels):
        self.gating.resample(labels)

    def resample_components(self, obs, labels):
        weights = one_hot(labels, K=self.size)
        self.components.resample(obs, weights)

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

    def meanfield_coordinate_descent(self, obs, randomize=True,
                                     maxiter=250, tol=1e-8,
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
                self.meanfield_update_parameters(obs, resp)
                resp = self.expected_responsibilities(obs)

                vlb.append(self.variational_lowerbound(obs, resp))

                if len(vlb) > 1:
                    if abs(vlb[-1] - vlb[-2]) < tol:
                        return vlb

                pbar.update(1)

        return vlb

    def meanfield_update_parameters(self, obs, resp):
        self.meanfield_update_components(obs, resp)
        self.meanfield_update_gating(resp)

    def meanfield_update_gating(self, resp):
        self.gating.meanfield_update(None, resp)

    def meanfield_update_components(self, obs, resp):
        self.components.meanfield_update(obs, resp)

    # SVI
    def meanfield_stochastic_descent(self, obs, randomize=True,
                                     maxiter=500, step_size=1e-2,
                                     batch_size=128, progress_bar=True,
                                     procces_id=0):

        vlb = []
        with tqdm(total=maxiter, desc=f'SVI #{procces_id + 1}',
                  position=procces_id, disable=not progress_bar) as pbar:

            scale = batch_size / float(len(obs))
            for i in range(maxiter):
                for batch in batches(batch_size, len(obs)):
                    if i == 0 and randomize is True:
                        resp = npr.rand(self.size, len(obs[batch, :]))
                        resp /= np.sum(resp, axis=0)
                    else:
                        resp = self.expected_responsibilities(obs[batch, :])

                    self.meanfield_sgdstep_parameters(obs[batch, :], resp,
                                                      scale, step_size)

                resp = self.expected_responsibilities(obs)
                vlb.append(self.variational_lowerbound(obs, resp))

                pbar.update(1)

        return vlb

    def meanfield_sgdstep_parameters(self, obs, resp, scale, step_size):
        self.meanfield_sgdstep_components(obs, resp, scale, step_size)
        self.meanfield_sgdstep_gating(resp, scale, step_size)

    def meanfield_sgdstep_components(self, obs, resp, scale, step_size):
        self.components.meanfield_sgdstep(obs, resp, scale, step_size)

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
        vlb += self.variational_lowerbound_obs(obs, resp)
        vlb += self.variational_lowerbound_labels(resp)
        return vlb

    def plot(self, obs, labels=None, color=None, legend=False, alpha=None):
        resp = self.expected_responsibilities(obs)
        labels = np.argmax(resp, axis=0)

        artists = self.likelihood.plot(obs, labels, color, legend, alpha)
        return artists
