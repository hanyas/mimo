import numpy as np
import numpy.random as npr

from mimo.distributions import LinearGaussianWithPrecision
from mimo.distributions import MatrixNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishart


# npr.seed(1337)

dcol = 50
drow = 1

A = 1. * npr.randn(drow, dcol)

nb_samples = 200
nb_datasets = 10

dist = LinearGaussianWithPrecision(A=A, lmbda=100 * np.eye(drow), affine=False)
x = [npr.randn(nb_samples, dcol) for _ in range(nb_datasets)]
y = [dist.rvs(_x) for _x in x]
print("True transf."+"\n", dist.A, "\n"+"True sigma"+"\n", dist.sigma)

affine = False
nb_params = dcol + 1 if affine else dcol

hypparams = dict(M=np.zeros((drow, nb_params)),
                 K=1e-2 * np.eye(nb_params),
                 psi=np.eye(drow),
                 nu=drow + 1)
prior = MatrixNormalWishart(**hypparams)

model = LinearGaussianWithMatrixNormalWishart(prior, affine=False)
model.meanfield_update(y=y, x=x)
print("ML transf."+"\n", model.likelihood.A, "\n"+"ML covariance"+"\n", model.likelihood.sigma)
