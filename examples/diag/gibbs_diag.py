import numpy as np
import numpy.random as npr

from mimo.distributions import GaussianWithDiagonalPrecision
from mimo.distributions import NormalGamma
from mimo.distributions import GaussianWithNormalGamma

# npr.seed(1337)

dim, nb_samples = 3, 500
dist = GaussianWithDiagonalPrecision(dim=dim, mu=npr.randn(dim),
                                     lmbda_diag=1. * npr.rand(dim))
data = dist.rvs(size=nb_samples)
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

hypparams = dict(dim=dim, mu=np.zeros((dim, )),
                 kappas=1e-2 * np.ones((dim, )),
                 alphas=(4. + 1e-8) / 2. * np.ones((dim, )),
                 betas=1. / (2. * np.ones((dim, ))))
prior = NormalGamma(**hypparams)

model = GaussianWithNormalGamma(dim=dim, prior=prior)
model.resample(data)
print("Gibbs mean"+"\n", model.likelihood.mu.T,
        "\n"+"Gibbs sigma"+"\n", model.likelihood.sigma)
