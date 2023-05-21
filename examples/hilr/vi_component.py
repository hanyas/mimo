import numpy as np
import numpy.random as npr

from mimo.distributions import Wishart
from mimo.distributions import NormalWishart

from mimo.distributions import MatrixNormalWithPrecision

from mimo.distributions import TiedGaussiansWithScaledPrecision
from mimo.distributions import TiedGaussiansWithHierarchicalNormalWisharts

from mimo.distributions import TiedAffineLinearGaussiansWithMatrixNormalWisharts

from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet

from mimo.mixtures import BayesianMixtureOfLinearGaussiansWithTiedActivation

import matplotlib.pyplot as plt


np.random.seed(1337)

# create data
nb_train = 999

x = np.linspace(-1.5, 1.5, nb_train)[:, None]

y = []
for i in range(3):
    t = np.linspace(-0.5, 0.5, int(nb_train / 3))[:, None]
    y.append(1. * t)
y = np.vstack(y)

y += 0.05 * npr.randn(nb_train).reshape(nb_train, 1)

y[:333] += 0.5
y[-333:] -= 0.5

z = np.hstack((0 * np.ones((int(nb_train / 3),), dtype=np.int32),
               1 * np.ones((int(nb_train / 3),), dtype=np.int32),
               2 * np.ones((int(nb_train / 3),), dtype=np.int32),))

plt.scatter(x, y, s=1.)
plt.show()

# learn model
gating_prior = Dirichlet(dim=3, alphas=np.ones((3, )))

gating = CategoricalWithDirichlet(dim=3, prior=gating_prior)

basis_hyper_prior = NormalWishart(dim=1,
                                  mu=np.zeros((1, )), kappa=1e-2,
                                  psi=np.eye(1), nu=2. + 1e-8)

basis_prior = TiedGaussiansWithScaledPrecision(size=3, dim=1,
                                               kappas=1e-2 * np.ones((3,)))

basis = TiedGaussiansWithHierarchicalNormalWisharts(size=3, dim=1,
                                                    hyper_prior=basis_hyper_prior,
                                                    prior=basis_prior)

slope_prior = MatrixNormalWithPrecision(column_dim=1, row_dim=1,
                                        M=np.zeros((1, 1)), K=1e-2 * np.eye(1))

offset_prior = TiedGaussiansWithScaledPrecision(size=3, dim=1,
                                                mus=np.zeros((3, 1)),
                                                kappas=1e-2 * np.ones((3,)))

precision_prior = Wishart(dim=1, psi=np.eye(1), nu=2. + 1e-16)

models = TiedAffineLinearGaussiansWithMatrixNormalWisharts(size=3, column_dim=1, row_dim=1,
                                                           slope_prior=slope_prior,
                                                           offset_prior=offset_prior,
                                                           precision_prior=precision_prior)

mixture = BayesianMixtureOfLinearGaussiansWithTiedActivation(size=3, input_dim=1,
                                                             output_dim=1, gating=gating,
                                                             basis=basis, models=models)

mixture.resample(x, y, maxiter=25, maxsubiter=25)

vlb = mixture.meanfield_coordinate_descent(x, y, randomize=True,
                                           maxiter=1000, maxsubiter=25, tol=1e-12)
print("vlb monoton?", np.all(np.diff(vlb) >= -1e-8))

print("VI transf."+"\n", mixture.models.likelihood.As,
      "\n"+"VI offset"+"\n", mixture.models.likelihood.cs,
      "\n"+"VI precision"+"\n", mixture.models.likelihood.lmbdas)
