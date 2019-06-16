import numpy as np
import numpy.random as npr

from mimo import distributions

dim, nb_samples, nb_datasets = 3, 500, 5
dist = distributions.Gaussian(mu=npr.randn(dim), sigma=5. * np.diag(npr.rand(dim)))
data = [dist.rvs(size=nb_samples) for _ in range(nb_datasets)]
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

hypparams = dict(mu=np.zeros((dim, )), kappa=0.05, psi=np.eye(dim), nu=2 * dim + 1)
prior = distributions.NormalInverseWishart(**hypparams)

model = distributions.BayesianGaussian(prior=prior)
model.meanfieldupdate(data)
print("Meanfield mean"+"\n", model.mu.T, "\n"+"Meanfield sigma"+"\n", model.sigma)
