import numpy as np
import numpy.random as npr

from mimo.distributions import Wishart

from mimo.distributions import GaussianWithScaledPrecision
from mimo.distributions import MatrixNormalWithPrecision

from mimo.distributions import AffineLinearGaussianWithMatrixNormalWishart


# npr.seed(1337)

column_dim = 49
row_dim = 1

A = 1. * npr.randn(row_dim, column_dim)
c = npr.randn(row_dim)

nb_samples = 2500

x = 10. * npr.randn(nb_samples, column_dim)
y = np.einsum('dl,nl->nd', A, x) + c + npr.randn(nb_samples, 1)

print("True transf."+"\n", A,
      "\n"+"True offset"+"\n", c)

slope_prior = MatrixNormalWithPrecision(column_dim=column_dim,
                                        row_dim=row_dim,
                                        M=np.zeros((row_dim, column_dim)),
                                        K=1e-2 * np.eye(column_dim))

offset_prior = GaussianWithScaledPrecision(dim=row_dim,
                                           mu=np.zeros((row_dim, )),
                                           kappa=1e-2)

precision_prior = Wishart(dim=row_dim, psi=np.eye(row_dim), nu=2. + 1e-8)

model = AffineLinearGaussianWithMatrixNormalWishart(column_dim=column_dim, row_dim=row_dim,
                                                    slope_prior=slope_prior,
                                                    offset_prior=offset_prior,
                                                    precision_prior=precision_prior)

model.resample(x, y, nb_iter=50)

print("Gibbs transf."+"\n", model.likelihood.A,
      "\n"+"Gibbs offset"+"\n", model.likelihood.c,
      "\n"+"Gibbs precision"+"\n", model.likelihood.lmbda)
