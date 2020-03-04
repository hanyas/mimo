import numpy as np
import numpy.random as npr

from mimo import distributions

npr.seed(1337)

in_dim = 30
out_dim = 1

_A = 1. * npr.randn(out_dim, in_dim)

nb_samples = 200
nb_datasets = 200

dist = distributions.LinearGaussian(A=_A, sigma=25e-2 * np.eye(out_dim), affine=False)
data = [dist.rvs(size=nb_samples) for _ in range(nb_datasets)]
print("True transf."+"\n", dist.A, "\n"+"True sigma"+"\n", dist.sigma)

affine = False
n_params = in_dim + 1 if affine else in_dim

hypparams = dict(M=np.zeros((out_dim, n_params)),
                 V=1. * np.eye(n_params),
                 affine=affine,
                 psi=np.eye(out_dim),
                 nu=2 * out_dim + 1)
prior = distributions.MatrixNormalInverseWishart(**hypparams)

model = distributions.BayesianLinearGaussian(prior)
model.resample(data)
print("Gibbs transf."+"\n", model.A, "\n"+"Gibbs sigma"+"\n", model.sigma)
