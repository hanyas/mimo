import numpy as np
import numpy.random as npr

from mimo import distributions

in_dim = 50
out_dim = 1

_A = 1. * npr.randn(out_dim, in_dim)

nb_samples = 200
nb_datasets = 200

dist = distributions.LinearGaussian(A=_A, sigma=2.5e-1 * np.eye(out_dim), affine=False)
data = [dist.rvs(size=nb_samples) for _ in range(nb_datasets)]
print("True transf."+"\n", dist.A, "\n"+"True sigma"+"\n", dist.sigma)

hypparams = dict(M=np.zeros((out_dim, in_dim)),
                 V=1. * np.eye(in_dim),
                 affine=False,
                 psi=np.eye(out_dim),
                 nu=2 * out_dim + 1)
prior = distributions.MatrixNormalInverseWishart(**hypparams)

model = distributions.BayesianLinearGaussian(prior)
model.meanfieldupdate(data)
print("Meanfield transf."+"\n", model.A, "\n"+"Meanfield sigma"+"\n", model.sigma)
