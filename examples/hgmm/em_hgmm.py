import numpy as np
import numpy.random as npr

from mimo.distributions import Categorical
from mimo.distributions import TiedGaussiansWithPrecision

from mimo.mixtures import MixtureOfGaussians

import matplotlib.pyplot as plt

# npr.seed(1337)

obs = []

# generate data
gating = Categorical(dim=2)

mus = np.stack([np.array([-7, -7.]),
                np.array([7., 7.])])

sigma = np.array([[3., 2.],
                  [2., 3.]])

lmbda = np.linalg.inv(sigma)
lmbdas = np.array(2 * [lmbda])

components = TiedGaussiansWithPrecision(size=2, dim=2,
                                        mus=mus, lmbdas=lmbdas)

gmm = MixtureOfGaussians(gating=gating, components=components)

_obs, _ = gmm.rvs(500)
obs.append(_obs)

plt.scatter(_obs[:, 0], _obs[:, 1], marker='+', color='blue')

#
gating = Categorical(dim=2)

mus = np.stack([np.array([7, -7.]),
                np.array([-7., 7.])])

sigma = np.array([[3., -2.],
                  [-2., 3.]])

lmbda = np.linalg.inv(sigma)
lmbdas = np.array(2 * [lmbda])

components = TiedGaussiansWithPrecision(size=2, dim=2,
                                        mus=mus, lmbdas=lmbdas)

gmm = MixtureOfGaussians(gating=gating, components=components)

_obs, _ = gmm.rvs(500)
obs.append(_obs)

plt.scatter(_obs[:, 0], _obs[:, 1], marker='+', color='red')

plt.xlim((-15., 15.))
plt.ylim((-15., 15.))
plt.show()


obs = np.vstack(obs)

from mimo.mixtures.hgmm import MixtureOfMixtureOfGaussians
from mimo.distributions import TiedGaussiansWithPrecision

cluster_size = 2
mixture_size = 2
dim = 2

gating = Categorical(dim=cluster_size)

components = []
for _ in range(cluster_size):
    _prob = npr.rand(mixture_size)
    _prob /= _prob.sum()
    _local_gating = Categorical(dim=mixture_size, probs=_prob)

    _mus = npr.randn(mixture_size, dim)
    _lmbdas = np.stack(mixture_size * [5. * np.eye(dim)])
    _local_components = TiedGaussiansWithPrecision(size=mixture_size, dim=dim,
                                                   mus=_mus, lmbdas=_lmbdas)

    _mixture = MixtureOfGaussians(gating=_local_gating,
                                  components=_local_components)
    components.append(_mixture)

model = MixtureOfMixtureOfGaussians(cluster_size=cluster_size,
                                    mixture_size=mixture_size, dim=dim,
                                    gating=gating, components=components)

ll = model.max_likelihood(obs, maxiter=2500, maxsubiter=5)
print("ll monoton?", np.all(np.diff(ll) >= -1e-8))

model.plot(obs)
