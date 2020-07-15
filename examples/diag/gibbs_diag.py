import numpy as np
import numpy.random as npr

from mimo.distributions import GaussianWithDiagonalCovariance
from mimo.distributions import NormalGamma
from mimo.distributions import GaussianWithNormalGamma

# npr.seed(1337)

dim, nb_samples, nb_datasets = 3, 500, 5
dist = GaussianWithDiagonalCovariance(mu=npr.randn(dim), sigmas=1. * npr.rand(dim))
data = [dist.rvs(size=nb_samples) for _ in range(nb_datasets)]
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)
#
hypparams = dict(mu=np.zeros((dim, )), kappas=1e-2 * np.ones((dim, )),
                 alphas=1. * np.ones((dim, )), betas=1. / 2. * np.ones((dim, )))
prior = NormalGamma(**hypparams)

model = GaussianWithNormalGamma(prior=prior)
model.resample(data)
print("Gibbs mean"+"\n", model.likelihood.mu.T, "\n"+"Gibbs sigma"+"\n", model.likelihood.sigma)
