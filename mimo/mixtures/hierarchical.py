import copy

import numpy as np
from scipy import special as special
from scipy.special import logsumexp

from mimo.abstraction import Distribution

from mimo.distributions.bayesian import CategoricalWithDirichlet
from mimo.distributions.bayesian import CategoricalWithStickBreaking

from mimo.util.decorate import pass_obs_arg, pass_obs_and_labels_arg
from mimo.util.stats import sample_discrete_from_log
from mimo.util.text import progprint_xrange

import pathos
nb_cores = pathos.multiprocessing.cpu_count()


class BayesianMixtureOfMixturesOfGaussians(Distribution):
    """
    This class is for mixtures of other distributions.
    """

    def __init__(self, gating, upper_components, lower_components):
        assert len(upper_components) > 0 and len(lower_components) > 0

        self.gating = gating
        self.upper_components = upper_components  # upper level density
        self.lower_components = lower_components  # lower level density

        self.obs = []
        self.labels = []

        self.whitend = False
        self.transform = None

    @property
    def nb_params(self):
        return self.gating.nb_params\
               + sum(uc.nb_params for uc in self.upper_components)\
               + sum(lc.nb_params for lc in self.lower_components)

    @property
    def size(self):
        return len(self.upper_components)

    @property
    def dim(self):
        return self.upper_components[0].dim

    @property
    def used_labels(self):
        assert self.has_data()
        label_usages = sum(np.bincount(_label, minlength=self.size) for _label in self.labels)
        used_labels, = np.where(label_usages > 0)
        return used_labels

    def add_data(self, obs, whiten=False, transform=False):

        obs = obs if isinstance(obs, list) else [obs]
        for _obs in obs:
            self.labels.append(self.gating.rvs(len(_obs)))

        if whiten:
            self.whitend = True

            if not transform:
                from sklearn.decomposition import PCA
                X = np.vstack([_obs for _obs in obs])
                self.transform = PCA(n_components=X.shape[-1], whiten=True)
                self.transform.fit(X)
            else:
                self.transform = transform

            for _obs in obs:
                self.obs.append(self.transform.transform(_obs))
        else:
            self.obs = obs

    def clear_data(self):
        self.obs.clear()
        self.labels.clear()

    def clear_transform(self):
        self.whitend = False
        self.transform = None

    def has_data(self):
        return len(self.obs) > 0

    def rvs(self, size=1):
        uz = self.gating.rvs(size)

        obs = np.zeros((size, self.dim))
        lz = np.zeros((size, self.dim))
        for idx, _uz in enumerate(uz):
            obs[idx, ...], lz[idx, ...] = self.lower_components[_uz].rvs()

        perm = np.random.permutation(size)
        obs, uz, lz = obs[perm], uz[perm], lz[perm]

        return obs, uz, lz

    def log_likelihood(self, obs):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def log_partition(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError