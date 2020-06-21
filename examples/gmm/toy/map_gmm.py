import numpy as np
import numpy.random as npr

from matplotlib import pyplot as plt

from mimo.distributions import Categorical
from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet
from mimo.distributions import GaussianWithCovariance
from mimo.distributions import NormalWishart
from mimo.distributions import GaussianWithNormalWishart

from mimo.mixtures import MixtureOfGaussians
from mimo.mixtures import BayesianMixtureOfGaussians

from mimo.util.text import progprint_xrange


npr.seed(1337)

gating = Categorical(K=2)

components = [GaussianWithCovariance(mu=np.array([1., 1.]), sigma=0.25 * np.eye(2)),
              GaussianWithCovariance(mu=np.array([-1., -1.]), sigma=0.5 * np.eye(2))]

gmm = MixtureOfGaussians(gating=gating, components=components)

obs, z = gmm.rvs(500)
gmm.plot(obs)

gating_hypparams = dict(K=2, alphas=np.ones((2, )))
gating_prior = Dirichlet(**gating_hypparams)

components_hypparams = dict(mu=np.zeros((2, )), kappa=0.01,
                            psi=np.eye(2), nu=3)
components_prior = NormalWishart(**components_hypparams)

model = BayesianMixtureOfGaussians(gating=CategoricalWithDirichlet(gating_prior),
                                   components=[GaussianWithNormalWishart(components_prior)
                                               for _ in range(2)])

model.add_data(obs)

print('Gibbs Sampling')
for _ in progprint_xrange(1000):
    model.max_aposteriori()

plt.figure()
model.plot(obs)
