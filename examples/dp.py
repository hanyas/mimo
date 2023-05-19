import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import stats

from mimo.distributions import TruncatedStickBreaking

alpha = 10
K = 1000

sb = TruncatedStickBreaking(K, np.ones(K), np.ones(K) * alpha)

weights = np.vstack([sb.rvs() for _ in range(10000)])
weights = np.mean(weights, axis=0)


import matplotlib.pyplot as plt

plt.axis([0, K + 1, 0, np.max(weights)])
plt.bar(range(1, K + 1), weights)
plt.show()


h = sc.stats.norm
omega = h.rvs(size=(10, K))

x = np.linspace(-3, 3, 200)
sample_cdfs = (weights[..., np.newaxis] * np.less.outer(omega, x)).sum(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, sample_cdfs[1:].T, c='gray', alpha=0.75)
ax.plot(x, h.cdf(x), c='k', label='Base CDF')
plt.show()
