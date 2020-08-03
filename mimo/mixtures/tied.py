import copy

import numpy as np
from scipy import special as special
from scipy.special import logsumexp

from mimo.abstraction import Distribution
from mimo.mixtures.full import MixtureOfGaussians

from mimo.distributions.bayesian import CategoricalWithDirichlet
from mimo.distributions.bayesian import CategoricalWithStickBreaking

from mimo.util.decorate import pass_obs_arg, pass_obs_and_labels_arg
from mimo.util.stats import sample_discrete_from_log
from mimo.util.text import progprint_xrange

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm
from pathos.helpers import mp


class MixtureOfTiedGaussians(MixtureOfGaussians):
    """
    This class is for mixtures of Gaussians
     sharing the same covariance matrix
    """

    def __init__(self, gating, ensemble):
        self.ensemble = ensemble
        super(MixtureOfTiedGaussians, self).__init__(gating=gating,
                                                     components=ensemble.components)

    @property
    def sigma(self):
        return self.ensemble.sigma

    @sigma.setter
    def sigma(self, value):
        self.ensemble.sigma = value

    @property
    def mus(self):
        return self.ensemble.mus

    @property
    def nb_params(self):
        return self.gating.nb_params + self.ensemble.nb_params

    def max_likelihood(self, obs, maxiter=1, progprint=True):

        current = mp.current_process()
        if len(current._identity) > 0:
            pos = current._identity[0] - 1
        else:
            pos = 0

        obs = obs if isinstance(obs, list) else [obs]

        with tqdm(total=maxiter, desc=f'EM #{pos + 1}',
                  position=pos, disable=not progprint) as pbar:
            for _ in range(maxiter):
                # Expectation step
                scores = [self.scores(_obs) for _obs in obs]

                # Maximization step
                self.ensemble.max_likelihood(obs, scores)
                self.gating.max_likelihood(None, scores)

                pbar.update(1)


class BayesianMixtureOfTiedGaussians(Distribution):
    """
    This class is for mixtures of other distributions.
    """

    def __init__(self, gating, ensemble):

        self.gating = gating
        self.ensemble = ensemble

        self.gaussians = self.ensemble.likelihood.components

        self.obs = []
        self.labels = []

        self.whitend = False
        self.transform = None

    @property
    def nb_params(self):
        return self.gating.likelihood.nb_params\
               + sum(g.nb_params for g in self.gaussians)\
               - (self.size - 1) * self.dim * (self.dim + 1) / 2

    @property
    def size(self):
        return self.ensemble.likelihood.size

    @property
    def dim(self):
        return self.ensemble.likelihood.dim

    @property
    def used_labels(self):
        assert self.has_data()
        label_usages = sum(np.bincount(_label, minlength=self.size)
                           for _label in self.labels)
        used_labels, = np.where(label_usages > 0)
        return used_labels

    def add_data(self, obs, whiten=False,
                 transform_type='PCA',
                 labels_from_prior=False):

        obs = obs if isinstance(obs, list) else [obs]

        if whiten:
            self.whitend = True

            data = np.vstack([_obs for _obs in obs])

            if transform_type == 'PCA':
                self.transform = PCA(n_components=data.shape[-1], whiten=True)
            elif transform_type == 'Standard':
                self.transform = StandardScaler()
            elif transform_type == 'MinMax':
                self.transform = MinMaxScaler((-1., 1.))
            else:
                raise NotImplementedError

            self.transform.fit(data)
            for _obs in obs:
                self.obs.append(self.transform.transform(_obs))
        else:
            self.obs = obs

        if labels_from_prior:
            for _obs in self.obs:
                self.labels.append(self.gating.likelihood.rvs(len(_obs)))
        else:
            self.labels = self._resample_labels(self.obs)

    def clear_data(self):
        self.obs.clear()
        self.labels.clear()

    def clear_transform(self):
        self.whitend = False
        self.transform = None

    def has_data(self):
        return len(self.obs) > 0

    def rvs(self, size=1):
        labels = self.gating.likelihood.rvs(size)
        counts = np.bincount(labels, minlength=self.size)

        obs = np.empty((size, self.dim))
        for idx, (g, count) in enumerate(zip(self.gaussians, counts)):
            obs[labels == idx, ...] = g.rvs(count)

        perm = np.random.permutation(size)
        obs, z = obs[perm], labels[perm]

        return obs, labels

    def log_likelihood(self, obs):
        assert isinstance(obs, (np.ndarray, list))
        if isinstance(obs, list):
            return sum(self.log_likelihood(_obs) for _obs in obs)
        else:
            scores = self.log_scores(obs)
            return np.sum(logsumexp(scores[~np.isnan(obs).any(axis=1)], axis=1))

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
        for idx, g in enumerate(self.gaussians):
            component_scores[:, idx] = g.log_likelihood(obs)
        component_scores = np.nan_to_num(component_scores, copy=False)

        gating_scores = self.gating.likelihood.log_likelihood(np.arange(K))
        score = gating_scores + component_scores
        return score

    def scores(self, obs):
        logr = self.log_scores(obs)
        score = np.exp(logr - np.max(logr, axis=1, keepdims=True))
        score /= np.sum(score, axis=1, keepdims=True)
        return score

    # Expectation-Maximization
    @pass_obs_arg
    def max_aposteriori(self, obs, maxiter=1, progprint=True):

        current = mp.current_process()
        if len(current._identity) > 0:
            pos = current._identity[0] - 1
        else:
            pos = 0

        obs = obs if isinstance(obs, list) else [obs]

        with tqdm(total=maxiter, desc=f'MAP #{pos + 1}',
                  position=pos, disable=not progprint) as pbar:
            for i in range(maxiter):
                # Expectation step
                scores = [self.scores(_obs) for _obs in obs]

                # Maximization step
                self.ensemble.max_aposteriori(obs, scores)
                self.gating.max_aposteriori(None, scores)

                pbar.update(1)

    # Gibbs sampling
    @pass_obs_and_labels_arg
    def resample(self, obs=None, labels=None,
                 maxiter=1, progprint=True):

        current = mp.current_process()
        if len(current._identity) > 0:
            pos = current._identity[0] - 1
        else:
            pos = 0

        with tqdm(total=maxiter, desc=f'Gibbs #{pos + 1}',
                  position=pos, disable=not progprint) as pbar:
            for _ in range(maxiter):
                self._resample_ensemble(obs, labels)
                self._resample_gating(labels)
                labels = self._resample_labels(obs)

                if self.has_data():
                    self.labels = labels

                pbar.update(1)

    def _resample_ensemble(self, obs, labels):
        self.ensemble.resample(data=obs, labels=labels)

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
        component_scores = self.ensemble.posterior.expected_log_likelihood(obs)
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
                    if np.abs(elbo[-1] - elbo[-2]) < tol:
                        return elbo
                pbar.update(1)

        # print('WARNING: meanfield_coordinate_descent hit maxiter of %d' % maxiter)
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
        self._meanfield_update_ensemble(obs, scores)
        self._meanfield_update_gating(scores)

    def _meanfield_update_gating(self, scores):
        self.gating.meanfield_update(None, scores)

    def _meanfield_update_ensemble(self, obs, scores):
        self.ensemble.meanfield_update(obs, scores)

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

    def _variational_lowerbound_obs(self, obs, scores):
        E_log_lik = self.ensemble.posterior.expected_log_likelihood(obs)
        return np.einsum('nk,nk->', E_log_lik, scores)

    def variational_lowerbound(self, obs, scores):
        vlb = 0.
        vlb += sum(self._variational_lowerbound_labels(_score) for _score in scores)
        vlb += self.gating.variational_lowerbound()
        vlb += self.ensemble.variational_lowerbound()
        vlb += sum(self._variational_lowerbound_obs(_obs, _score)
                   for _obs, _score in zip(obs, scores))

        # add in symmetry factor (if we're actually symmetric)
        if len(set(type(c) for c in self.ensemble.posterior.components)) == 1:
            vlb += special.gammaln(self.size + 1)

        return vlb

    @pass_obs_arg
    def plot(self, obs=None, color=None, legend=False, alpha=None):
        # I haven't implemented plotting
        # for whitend data, it's a hassle :D
        assert self.whitend is False

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

        labels = []
        for _obs in obs:
            labels.append(np.argmax(self.scores(_obs), axis=1))

        # plot data scatter
        for _obs, _label in zip(obs, labels):
            colorseq = [label_colors[l] for l in _label]
            artists.append(plt.scatter(_obs[:, 0], _obs[:, 1], c=colorseq, marker='+'))

        # plot parameters
        axis = plt.axis()
        for label, (g, w) in enumerate(zip(self.gaussians, self.gating.likelihood.probs)):
            artists.extend(g.plot(color=label_colors[label], label='%d' % label,
                                  alpha=min(0.25, 1. - (1. - w) ** 2) / 0.25 if alpha is None else alpha))
        plt.axis(axis)

        # add legend
        if legend and color is None:
            plt.legend([plt.Rectangle((0, 0), 1, 1, fc=c)
                        for i, c in label_colors.items() if i in self.used_labels],
                       [i for i in label_colors if i in self.used_labels], loc='best', ncol=2)
        plt.show()

        return artists