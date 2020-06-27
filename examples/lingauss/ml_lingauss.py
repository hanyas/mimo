import numpy as np
import numpy.random as npr

from mimo.distributions import LinearGaussianWithPrecision


npr.seed(1337)

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

model = LinearGaussianWithPrecision(affine=False)
model.max_likelihood(y=y, x=x)
print("ML transf."+"\n", model.A, "\n"+"ML covariance"+"\n", model.sigma)
