import numpy as np
import numpy.random as npr

from mimo import distributions

npr.seed(1337)

in_dim = 50
out_dim = 1

_A = 1. * npr.randn(out_dim, in_dim)

nb_samples = 200
nb_datasets = 200

dist = distributions.LinearGaussian(A=_A, sigma=25e-2 * np.eye(out_dim), affine=False)
x = [npr.randn(nb_samples, in_dim) for _ in range(nb_datasets)]
y = [dist.rvs(_x) for _x in x]
print("True transf."+"\n", dist.A, "\n"+"True sigma"+"\n", dist.sigma)

affine = False
n_params = in_dim + 1 if affine else in_dim

hypparams = dict(M=np.zeros((out_dim, n_params)),
                 V=1. * np.eye(n_params),
                 affine=affine,
                 psi=np.eye(out_dim),
                 nu=2 * out_dim + 1)
prior = distributions.MatrixNormalInverseWishart(**hypparams)

model = distributions.LinearGaussianWithMatrixNormalInverseWishart(prior)
model.meanfield_update(y=y, x=x)
print("Meanfield transf."+"\n", model.A, "\n"+"Meanfield sigma"+"\n", model.sigma)
