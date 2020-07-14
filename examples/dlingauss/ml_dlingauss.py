import numpy as np
import numpy.random as npr

from mimo.distributions import LinearGaussianWithDiagonalPrecision


npr.seed(1337)

dcol = 10
drow = 3

A = 1. * npr.randn(drow, dcol)

nb_samples = 200
nb_datasets = 10

dist = LinearGaussianWithDiagonalPrecision(A=A, lmbdas=100. * np.ones(drow), affine=False)
x = [npr.randn(nb_samples, dcol) for _ in range(nb_datasets)]
y = [dist.rvs(_x) for _x in x]
print("True transf."+"\n", dist.A, "\n"+"True sigma"+"\n", dist.sigmas)

affine = False

model = LinearGaussianWithDiagonalPrecision(affine=False)
model.max_likelihood(y=y, x=x)
print("ML transf."+"\n", model.A, "\n"+"ML covariance"+"\n", model.sigmas)
