import numpy as np
import numpy.random as npr

from mimo.distributions import GaussianWithCovariance
from mimo.distributions import GaussianWithKnownScaledPrecision
from mimo.distributions import NormalWishart
from mimo.distributions import GaussianWithHierarchicalNormalWishart

# npr.seed(1337)

dim, nb_samples = 3, 1000
dist = GaussianWithCovariance(dim=dim, mu=npr.randn(dim),
                              sigma=2. * np.eye(dim))
data = dist.rvs(size=nb_samples)
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

prior = GaussianWithKnownScaledPrecision(dim=dim, kappa=1.)
hyper_prior = NormalWishart(dim=dim, mu=np.zeros((dim, )),
                            kappa=1e-2, psi=np.eye(dim), nu=dim + 1 + 1e-8)

model = GaussianWithHierarchicalNormalWishart(dim=dim, prior=prior,
                                              hyper_prior=hyper_prior)

vlb = model.meanfield_update(data, nb_iter=25)
print("ML mean"+"\n", model.likelihood.mu.T,
      "\n"+"ML sigma"+"\n", model.likelihood.sigma)
