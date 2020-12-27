import numpy as np
import numpy.random as npr

from mimo.distributions import GaussianWithCovariance
from mimo.distributions import GaussianWithPrecision
from mimo.distributions import GaussianWithKnownPrecision
from mimo.distributions import GaussianWithNormal

npr.seed(1337)

dim, nb_samples, nb_datasets = 3, 500, 5
dist = GaussianWithCovariance(mu=npr.randn(dim), sigma=1. * np.diag(npr.rand(dim)))
data = [dist.rvs(size=nb_samples) for _ in range(nb_datasets)]
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

hypparams = dict(mu=np.zeros((dim, )), lmbda=np.eye(dim) * 1e-6)
prior = GaussianWithPrecision(**hypparams)

likelihood = GaussianWithKnownPrecision(lmbda=dist.lmbda)
model = GaussianWithNormal(prior=prior, likelihood=likelihood)
model.meanfield_update(data)
print("Meanfield mean"+"\n", model.posterior.mu.T, "\n"+"Meanfield sigma"+"\n", model.posterior.sigma)
