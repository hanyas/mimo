import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

from mimo import distributions
from mimo import mixtures

from mimo.util.text import progprint_xrange

npr.seed(1337)


gating = distributions.Categorical(K=2)
components = [distributions.Gaussian(mu=np.array([1., 1.]), sigma=0.25 * np.eye(2)),
              distributions.Gaussian(mu=np.array([-1., -1.]), sigma=0.5 * np.eye(2))]

gmm = mixtures.full.MixtureOfGaussians(gating=gating, components=components)

obs, z = gmm.rvs(1000)
gmm.plot(obs)

gating = distributions.Categorical(K=2)
components = [distributions.Gaussian(mu=npr.randn(2,) * 2, sigma=np.eye(2)) for _ in range(2)]

model = mixtures.full.MixtureOfGaussians(gating=gating, components=components)

print('Expecation Maximization')
for _ in progprint_xrange(500):
    model.max_likelihood(obs)

plt.figure()
model.plot(obs)
