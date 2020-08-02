import numpy as np
import numpy.random as npr

from matplotlib import pyplot as plt

from mimo.distributions import Categorical
from mimo.distributions import GaussianWithCovariance
from mimo.distributions import GaussianWithPrecision

from mimo.mixtures import MixtureOfGaussians


npr.seed(1337)

gating = Categorical(K=2)

components = [GaussianWithCovariance(mu=np.array([1., 1.]), sigma=0.25 * np.eye(2)),
              GaussianWithCovariance(mu=np.array([-1., -1.]), sigma=0.5 * np.eye(2))]

gmm = MixtureOfGaussians(gating=gating, components=components)

obs, z = gmm.rvs(500)
gmm.plot(obs)

gating = Categorical(K=2)
components = [GaussianWithPrecision(mu=npr.randn(2, ),
                                    lmbda=np.eye(2)) for _ in range(2)]

model = MixtureOfGaussians(gating=gating, components=components)

model.max_likelihood(obs, maxiter=500)

plt.figure()
model.plot(obs)
