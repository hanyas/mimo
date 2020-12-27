import numpy as np
import numpy.random as npr

from mimo.distributions import LinearGaussianWithPrecision
from mimo.distributions import LinearGaussianWithKnownPrecision
from mimo.distributions import MatrixNormalWithKnownPrecision
from mimo.distributions import LinearGaussianWithMatrixNormal


# npr.seed(1337)

dcol = 5
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
                 K=1e-6 * np.eye(nb_params),
                 V=dist.lmbda)
prior = MatrixNormalWithKnownPrecision(**hypparams)

likelihood = LinearGaussianWithKnownPrecision(lmbda=dist.lmbda, affine=dist.affine)
model = LinearGaussianWithMatrixNormal(prior, likelihood=likelihood, affine=dist.affine)
model.meanfield_update(y=y, x=x)
print("VI transf."+"\n", model.posterior.M, "\n"+"VI covariance"+"\n", model.posterior.K)
