import numpy as np
import numpy.random as npr

from matplotlib import pyplot as plt
from mimo import distributions

in_dim = 1
out_dim = 1

_A = 1. * npr.randn(out_dim, in_dim)

nb_samples = 4000
affine_data = False
affine_model = True

dist1 = distributions.Gaussian(mu=npr.randn(in_dim), sigma=5. * np.diag(npr.rand(in_dim)))
x = dist1.rvs(size=nb_samples)
x = np.asarray(x)

dist2 = distributions.LinearGaussian(A=_A, sigma=2.5e-1 * np.eye(out_dim), affine=affine_data)

data = dist2.rvs(size=nb_samples, x=x)

# # plot data
# plt.figure()
# plt.plot(data[:, 0], data[:, 1], 'kx')
# plt.title('Data')
# plt.show()

# if affine:
#     out_dim = out_dim + 1
hypparams = dict(mu=np.zeros((in_dim,)),
                 kappa=0.05,
                 psi_niw=np.eye(in_dim),
                 nu_niw=2 * in_dim + 1,
                 M=np.zeros((out_dim, in_dim)),
                 V=1. * np.eye(in_dim),
                 affine= affine_model,
                 psi_mniw=np.eye(out_dim),
                 nu_mniw=2 * out_dim + 1)


prior = distributions.NormalInverseWishartMatrixNormalInverseWishart(**hypparams)

model = distributions.BayesianLinearGaussianWithNoisyInputs(prior)
model = model.max_likelihood(data)
print("True transf. x"+"\n", dist1.mu, "\n"+"True covariance x"+"\n", dist1.sigma)
print("ML transf. x"+"\n", model.mu, "\n"+"ML covariance x"+"\n", model.sigma_niw)
print("------------------------------------------------------------------------------------------------")
print("True transf. y"+"\n", dist2.A, "\n"+"True covariance y"+"\n", dist2.sigma)
print("ML transf. y"+"\n", model.A, "\n"+"ML covariance y"+"\n", model.sigma)

