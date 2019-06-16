import numpy as np
import numpy.random as npr

from mimo import distributions

dim, nb_samples, nb_datasets = 3, 200, 5
dist = distributions.DiagonalGaussian(mu=npr.randn(dim), sigmas=5. * npr.rand(dim))
data = [dist.rvs(size=nb_samples) for _ in range(nb_datasets)]
print("True mean" + "\n", dist.mu.T, "\n" + "True sigma" + "\n", dist.sigma)

hypparams = dict(mu=np.zeros((dim, )), kappas=0.05 * np.ones((dim, )),
                 alphas=np.ones((dim, )), betas=2. * np.ones((dim, )))
prior = distributions.NormalInverseGamma(**hypparams)

model = distributions.BayesianDiagonalGaussian(prior=prior)
model.resample(data)
print("Gibbs mean"+"\n", model.mu.T, "\n"+"Gibbs sigma"+"\n", model.sigma)
