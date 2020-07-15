import numpy as np
import numpy.random as npr

from matplotlib import pyplot as plt

from mimo.distributions import Categorical
from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet
from mimo.distributions import GaussianWithCovariance
from mimo.distributions import NormalGamma
from mimo.distributions import GaussianWithNormalGamma

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

components_hypparams = dict(mu=np.zeros((2, )),
                            kappas=1e-2 * np.ones((2, )),
                            alphas=1. * np.ones((2, )),
                            betas=1. / (2. * 1e4) * np.ones((2, )))
components_prior = NormalGamma(**components_hypparams)

model = BayesianMixtureOfGaussians(gating=CategoricalWithDirichlet(gating_prior),
                                   components=[GaussianWithNormalGamma(components_prior)
                                               for _ in range(2)])

model.add_data(obs)

model.resample()
print('Variational Inference')
for _ in progprint_xrange(2500):
    model.meanfield_update()

plt.figure()
model.plot(obs)
