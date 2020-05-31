import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

from mimo import distributions, mixture
from mimo.util.text import progprint_xrange

npr.seed(1337)

nb_samples = 2500

data = np.zeros((nb_samples, 2))
step = 14. * np.pi / nb_samples

for i in range(data.shape[0]):
    x = i * step - 6.
    data[i, 0] = x + npr.normal(0, 0.1)
    data[i, 1] = 3. * (np.sin(x) + npr.normal(0, .1))

plt.figure()
plt.plot(data[:, 0], data[:, 1], 'kx')
plt.title('data')

nb_models = 25

gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, )))
gating_prior = distributions.Dirichlet(**gating_hypparams)

components_hypparams = dict(mu=np.mean(data, axis=0), kappa=0.01, psi=np.eye(2), nu=3)
components_prior = distributions.NormalInverseWishart(**components_hypparams)

gmm = mixture.BayesianMixtureOfGaussians(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                                         components=[distributions.BayesianGaussian(components_prior) for _ in range(nb_models)])

gmm.add_data(data)

print('Expecation Maximization')
gmm.resample()
for _ in progprint_xrange(1000):
    gmm.max_likelihood()

plt.figure()
gmm.plot()
plt.title('posterior')
plt.show()
