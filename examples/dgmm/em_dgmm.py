import numpy as np
import numpy.random as npr

from matplotlib import pyplot as plt

from mimo.distributions import Categorical
from mimo.distributions import GaussianWithDiagonalCovariance
from mimo.distributions import GaussianWithDiagonalPrecision

from mimo.mixtures import MixtureOfGaussians


npr.seed(1337)

gating = Categorical(K=2)

components = [GaussianWithDiagonalCovariance(mu=np.array([1., 1.]), sigmas=np.array([0.25, 0.5])),
              GaussianWithDiagonalCovariance(mu=np.array([-1., -1.]), sigmas=np.array([0.5, 0.25]))]

gmm = MixtureOfGaussians(gating=gating, components=components)

obs, z = gmm.rvs(500)
gmm.plot(obs)

gating = Categorical(K=2)
components = [GaussianWithDiagonalPrecision(mu=npr.randn(2, ),
                                            lmbdas=np.ones((2, ))) for _ in range(2)]

model = MixtureOfGaussians(gating=gating, components=components)

print('Expecation Maximization')
model.max_likelihood(obs, maxiter=500)

plt.figure()
model.plot(obs)
