import numpy as np
import numpy.random as npr

from mimo.distributions import GaussianWithDiagonalCovariance
from mimo.distributions import NormalGamma
from mimo.distributions import GaussianWithNormalGamma

npr.seed(1337)

dim, nb_samples, nb_datasets = 3, 500, 5
dist = GaussianWithDiagonalCovariance(mu=npr.randn(dim), sigmas=1. * npr.rand(dim))
data = [dist.rvs(size=nb_samples) for _ in range(nb_datasets)]
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

model = GaussianWithDiagonalCovariance(mu=np.zeros((dim, )))
model.max_likelihood(data)
print("ML mean"+"\n", model.mu.T, "\n"+"ML sigma"+"\n", model.sigma)

hypparams = dict(mu=np.zeros((dim, )), kappas=0.05 * np.ones((dim, )),
                 alphas=np.ones((dim, )), betas=2. * np.ones((dim, )))
prior = NormalGamma(**hypparams)

model = GaussianWithNormalGamma(prior=prior)
model.max_aposteriori(data)
print("ML mean"+"\n", model.likelihood.mu.T, "\n"+"ML sigma"+"\n", model.likelihood.sigma)
