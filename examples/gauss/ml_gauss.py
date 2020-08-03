import numpy as np
import numpy.random as npr

from mimo.distributions import GaussianWithCovariance
from mimo.distributions import GaussianWithPrecision
from mimo.distributions import NormalWishart
from mimo.distributions import GaussianWithNormalWishart

npr.seed(1337)

dim, nb_samples, nb_datasets = 3, 500, 5
dist = GaussianWithCovariance(mu=npr.randn(dim), sigma=1. * np.diag(npr.rand(dim)))
data = [dist.rvs(size=nb_samples) for _ in range(nb_datasets)]
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

model = GaussianWithPrecision(mu=np.zeros((dim, )))
model.max_likelihood(data)
print("ML mean"+"\n", model.mu.T, "\n"+"ML sigma"+"\n", model.sigma)


hypparams = dict(mu=np.zeros((dim, )), kappa=0.01, psi=np.eye(dim), nu=dim + 1)
prior = NormalWishart(**hypparams)

model = GaussianWithNormalWishart(prior=prior)
model.max_aposteriori(data)
print("MAP mean"+"\n", model.likelihood.mu.T, "\n"+"MAP sigma"+"\n", model.likelihood.sigma)
