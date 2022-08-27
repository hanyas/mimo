import numpy as np
import numpy.random as npr

from mimo.distributions import LinearGaussianWithDiagonalPrecision


# npr.seed(1337)

column_dim = 50
row_dim = 3

A = 1. * npr.randn(row_dim, column_dim)

nb_samples = 2500

dist = LinearGaussianWithDiagonalPrecision(column_dim, row_dim,
                                           A=A, lmbda_diag=10. * np.ones((row_dim, )),
                                           affine=True)
x = npr.randn(nb_samples, column_dim - 1)
y = dist.rvs(x)

print("True transf."+"\n", dist.A,
      "\n"+"True precision"+"\n", dist.lmbda)

model = LinearGaussianWithDiagonalPrecision(column_dim, row_dim,
                                            affine=True)
model.max_likelihood(x=x, y=y)
print("ML transf."+"\n", model.A,
      "\n"+"ML precision"+"\n", model.lmbda)
