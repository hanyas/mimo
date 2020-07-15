import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from mimo.distributions import MatrixNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishartAndAutomaticRelevance
from mimo.distributions import Gamma

# np.random.seed(1337)

nb_samples, nb_features = 100, 100

X = np.random.randn(nb_samples, nb_features)

lambda_ = 1e-2
w = np.zeros(nb_features)

relevant_features = np.random.randint(0, nb_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0., scale=1. / np.sqrt(lambda_))

alpha_ = 10.
noise = stats.norm.rvs(loc=0., scale=1. / np.sqrt(alpha_), size=nb_samples)

y = np.dot(X, w) + noise
y = y.reshape(-1, 1)

hypparams = dict(M=np.zeros((1, nb_features)),
                 K=np.eye(nb_features),
                 psi=np.eye(1),
                 nu=1 + 2)
prior = MatrixNormalWishart(**hypparams)

std = LinearGaussianWithMatrixNormalWishart(prior, affine=False)
std.resample(y=y, x=X)
print("Gibbs transf."+"\n", std.likelihood.A, "\n"+"Gibbs covariance"+"\n", std.likelihood.sigma)

hyphypparams = dict(alphas=1. * np.ones(nb_features),
                    betas=1. / 1e2 * np.ones(nb_features))

hypprior = Gamma(**hyphypparams)

ard = LinearGaussianWithMatrixNormalWishartAndAutomaticRelevance(prior, hypprior, affine=False)
ard.resample(y=y, x=X)
print("ARD transf."+"\n", ard.likelihood.A, "\n"+"ARD transf covariance"+"\n", ard.likelihood.sigma)

plt.figure(figsize=(6, 5))
plt.title("Weights of the model")
plt.plot(w, color='orange', linestyle='-', linewidth=2, label="Ground truth")
plt.plot(std.likelihood.A.flatten(), color='darkblue', linestyle='-', linewidth=2, label="STD estimate")
plt.plot(ard.likelihood.A.flatten(), color='green', linestyle='-', linewidth=2, label="ARD estimate")
plt.legend(('true', 'standard', 'ard'))
