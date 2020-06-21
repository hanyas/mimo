import numpy as np
import numpy.random as npr

from mimo import distributions

npr.seed(1337)

dim, nb_samples, nb_datasets = 3, 500, 5
dist = distributions.GaussianWithCovariance(mu=npr.randn(dim), sigma=1. * np.diag(npr.rand(dim)))
data = [dist.rvs(size=nb_samples) for _ in range(nb_datasets)]
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

hypparams = dict(mu=np.zeros((dim, )), kappa=0.01, psi=np.eye(dim), nu=dim + 1)
prior = distributions.NormalWishart(**hypparams)

model = distributions.GaussianWithNormalWishart(prior=prior)
model.resample(data)
print("Gibbs mean"+"\n", model.likelihood.mu.T, "\n"+"Gibbs sigma"+"\n", model.likelihood.sigma)
