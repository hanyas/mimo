import numpy as np
import numpy.random as npr

from matplotlib import pyplot as plt

from mimo.distributions import Categorical
from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet
from mimo.distributions import GaussianWithDiagonalCovariance
from mimo.distributions import NormalGamma
from mimo.distributions import GaussianWithNormalGamma

from mimo.mixtures import MixtureOfGaussians
from mimo.mixtures import BayesianMixtureOfGaussians

from mimo.util.text import progprint_xrange


npr.seed(1337)

gating = Categorical(K=2)

components = [GaussianWithDiagonalCovariance(mu=np.array([1., 1.]), sigmas=np.array([0.25, 0.5])),
              GaussianWithDiagonalCovariance(mu=np.array([-1., -1.]), sigmas=np.array([0.5, 0.25]))]

gmm = MixtureOfGaussians(gating=gating, components=components)

obs, z = gmm.rvs(500)
gmm.plot(obs)

gating_hypparams = dict(K=2, alphas=np.ones((2, )))
gating_prior = Dirichlet(**gating_hypparams)

components_hypparams = dict(mu=np.zeros((2, )),
                            kappas=1e-2 * np.ones((2, )),
                            alphas=1. * np.ones((2, )),
                            betas=1. / (2. * 1e4) * np.ones((2, )))
components_prior = NormalGamma(**components_hypparams)

model = BayesianMixtureOfGaussians(gating=CategoricalWithDirichlet(gating_prior),
                                   components=[GaussianWithNormalGamma(components_prior)
                                               for _ in range(2)])

model.add_data(obs)

print('Gibbs Sampling')
for _ in progprint_xrange(1000):
    model.max_aposteriori()

plt.figure()
model.plot(obs)
