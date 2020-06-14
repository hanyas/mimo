import numpy as np
import numpy.random as npr

from scipy import stats

from matplotlib import pyplot as plt

from mimo import distributions
from mimo import mixtures

from mimo.util.text import progprint_xrange

# npr.seed(1337)

gating = distributions.Categorical(K=2)

sigma = stats.invwishart(3, np.eye(2)).rvs()
ensemble = distributions.TiedGaussians(mus=[np.array([1., 1.]),
                                            np.array([-1., -1.])],
                                       sigma=sigma)

gmm = mixtures.MixtureOfTiedGaussians(gating=gating, ensemble=ensemble)

obs, z = gmm.rvs(500)
gmm.plot(obs)

gating = distributions.Categorical(K=2)
ensemble = distributions.TiedGaussians(mus=[npr.randn(2, ),
                                            npr.randn(2,)],
                                       sigma=np.eye(2))

model = mixtures.MixtureOfTiedGaussians(gating=gating, ensemble=ensemble)

print('Expecation Maximization')
for _ in progprint_xrange(1000):
    model.max_likelihood(obs)

plt.figure()
model.plot(obs)
