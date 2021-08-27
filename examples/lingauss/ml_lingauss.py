import numpy as np
import numpy.random as npr

from mimo.distributions import LinearGaussianWithPrecision


# npr.seed(1337)

column_dim = 50
row_dim = 1

A = 1. * npr.randn(row_dim, column_dim)
affine = False

nb_samples = 2500

dist = LinearGaussianWithPrecision(column_dim, row_dim,
                                   A=A, lmbda=10. * np.eye(row_dim),
                                   affine=affine)
x = npr.randn(nb_samples, column_dim)
y = dist.rvs(x)

print("True transf."+"\n", dist.A,
      "\n"+"True precision"+"\n", dist.lmbda)

model = LinearGaussianWithPrecision(column_dim, row_dim,
                                    affine=affine)
model.max_likelihood(x=x, y=y)
print("ML transf."+"\n", model.A,
      "\n"+"ML precision"+"\n", model.lmbda)
