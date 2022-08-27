import numpy as np
import numpy.random as npr

from mimo.distributions import LinearGaussianWithPrecision
from mimo.distributions import AffineLinearGaussianWithPrecision


npr.seed(1337)

column_dim = 50
row_dim = 1

A = 1. * npr.randn(row_dim, column_dim)

nb_samples = 2500

dist = LinearGaussianWithPrecision(column_dim, row_dim,
                                   A=A, lmbda=10. * np.eye(row_dim),
                                   affine=True)
x = npr.randn(nb_samples, column_dim - 1)
y = dist.rvs(x)

print("True transf."+"\n", dist.A[:, :-1],
      "\n"+"True offset"+"\n", dist.A[:, -1],
      "\n"+"True precision"+"\n", dist.lmbda)

model = AffineLinearGaussianWithPrecision(column_dim - 1, row_dim)

model.max_likelihood(x=x, y=y, nb_iter=5)
print("ML transf."+"\n", model.A,
      "\n"+"ML offset"+"\n", model.c,
      "\n"+"ML precision"+"\n", model.lmbda)
