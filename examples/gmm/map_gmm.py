import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

from mimo import distributions
from mimo import mixtures

from mimo.util.text import progprint_xrange

npr.seed(1337)


gating = distributions.Categorical(K=2)
components = [distributions.Gaussian(mu=np.array([1., 1.]), sigma=0.25 * np.eye(2)),
              distributions.Gaussian(mu=np.array([-1., -1.]), sigma=0.5 * np.eye(2))]

gmm = mixtures.MixtureOfGaussians(gating=gating, components=components)

obs, z = gmm.rvs(1000)
gmm.plot(obs)

gating_hypparams = dict(K=2, alphas=np.ones((2, )))
gating_prior = distributions.Dirichlet(**gating_hypparams)

components_hypparams = dict(mu=np.zeros((2, )), kappa=0.01, psi=np.eye(2), nu=3)
components_prior = distributions.NormalInverseWishart(**components_hypparams)

model = mixtures.BayesianMixtureOfGaussians(gating=distributions.CategoricalWithDirichlet(gating_prior),
                                            components=[distributions.GaussianWithNormalInverseWishart(components_prior)
                                                        for _ in range(2)])

model.add_data(obs)

model.resample()
print('Expecation Maximization')
for _ in progprint_xrange(500):
    model.max_aposteriori()

plt.figure()
model.plot(obs)
