import numpy as np
import numpy.random as npr

from mimo.distributions import GaussianWithCovariance
from mimo.distributions import GaussianWithPrecision
from mimo.distributions import NormalWishart
from mimo.distributions import GaussianWithNormalWishart

# npr.seed(1337)

dim, nb_samples = 3, 1000
dist = GaussianWithCovariance(dim=dim, mu=npr.randn(dim),
                              sigma=1. * np.diag(npr.rand(dim)))
data = dist.rvs(size=nb_samples)
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

model = GaussianWithPrecision(dim=dim, mu=np.zeros((dim, )))
model.max_likelihood(data)
print("ML mean"+"\n", model.mu.T, "\n"+"ML sigma"+"\n", model.sigma)

hypparams = dict(dim=dim, mu=np.zeros((dim, )),
                 kappa=0.01, psi=np.eye(dim), nu=dim + 1)
prior = NormalWishart(**hypparams)

model = GaussianWithNormalWishart(dim=dim, prior=prior)
model.max_aposteriori(data)
print("MAP mean"+"\n", model.likelihood.mu.T, "\n"+"MAP sigma"+"\n", model.likelihood.sigma)
