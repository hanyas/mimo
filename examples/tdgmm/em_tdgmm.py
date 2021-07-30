import numpy as np
import numpy.random as npr

from mimo.distributions import Categorical
from mimo.distributions import StackedGaussiansWithPrecision
from mimo.distributions import TiedGaussiansWithDiagonalPrecision

from mimo.mixtures import MixtureOfGaussians

from matplotlib import pyplot as plt


# npr.seed(1337)

# generate data
gating = Categorical(dim=4)

mus = np.stack([np.array([-3., 3.]),
                np.array([3., -3.]),
                np.array([5., 5.]),
                np.array([-5., -5.])])

lmbdas = np.stack([4. * np.eye(2),
                   3. * np.eye(2),
                   2. * np.eye(2),
                   1. * np.eye(2)])

components = StackedGaussiansWithPrecision(size=4, dim=2,
                                           mus=mus, lmbdas=lmbdas)

gmm = MixtureOfGaussians(gating=gating, components=components)

obs, labels = gmm.rvs(500)
gmm.plot(obs)

# learn model
gating = Categorical(dim=4)

mus = npr.randn(4, 2)
lmbdas_diags = np.ones((4, 2))

components = TiedGaussiansWithDiagonalPrecision(size=4, dim=2,
                                                   mus=mus, lmbdas_diags=lmbdas_diags)

model = MixtureOfGaussians(gating=gating, components=components)

ll = model.max_likelihood(obs, maxiter=100)
print("ll monoton?", np.all(np.diff(ll) >= -1e-8))

plt.figure()
model.plot(obs)
