import numpy as np
import numpy.random as npr

from mimo.distributions import GaussianWithCovariance
from mimo.distributions import NormalWishart
from mimo.distributions import GaussianWithNormalWishart

# npr.seed(1337)

dim, nb_samples = 3, 1000
dist = GaussianWithCovariance(dim=dim, mu=npr.randn(dim),
                              sigma=1. * np.diag(npr.rand(dim)))
data = dist.rvs(size=nb_samples)
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

hypparams = dict(dim=dim, mu=np.zeros((dim, )),
                 kappa=0.01, psi=np.eye(dim), nu=dim + 1)
prior = NormalWishart(**hypparams)

model = GaussianWithNormalWishart(dim=dim, prior=prior)
model.meanfield_update(data)
print("Meanfield mean"+"\n", model.likelihood.mu.T,
      "\n"+"Meanfield sigma"+"\n", model.likelihood.sigma)
