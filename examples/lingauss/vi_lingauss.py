import numpy as np
import numpy.random as npr

from mimo.distributions import LinearGaussianWithPrecision
from mimo.distributions import MatrixNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishart


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

nb_params = column_dim + 1 if affine else column_dim

hypparams = dict(column_dim=column_dim,
                 row_dim=row_dim,
                 M=np.zeros((row_dim, nb_params)),
                 K=1e-2 * np.eye(nb_params),
                 psi=np.eye(row_dim),
                 nu=row_dim + 1)
prior = MatrixNormalWishart(**hypparams)

model = LinearGaussianWithMatrixNormalWishart(column_dim, row_dim,
                                              prior, affine=affine)
model.meanfield_update(x=x, y=y)
print("Gibbs transf."+"\n", model.likelihood.A,
      "\n"+"Gibbs precision"+"\n", model.likelihood.lmbda)
