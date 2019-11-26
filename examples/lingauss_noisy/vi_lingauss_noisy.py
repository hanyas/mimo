import numpy as np
import numpy.random as npr

from mimo import distributions

in_dim = 2
out_dim = 1

_A = 1. * npr.randn(out_dim, in_dim)

nb_samples = 4000

input_dist = distributions.Gaussian(mu=npr.randn(in_dim), sigma=5. * np.diag(npr.rand(in_dim)))
x = input_dist.rvs(size=nb_samples)
x = np.asarray(x)

output_dist = distributions.LinearGaussian(A=_A, sigma=25e-2 * np.eye(out_dim), affine=False)
data = output_dist.rvs(size=nb_samples, x=x)

affine_model = False
if affine_model:
    n_params = in_dim + 1
else:
    n_params = in_dim

hypparams = dict(mu=np.zeros((in_dim,)),
                 kappa=0.05,
                 psi_niw=np.eye(in_dim),
                 nu_niw=2 * in_dim + 1,
                 M=np.zeros((out_dim, n_params)),
                 V=1. * np.eye(n_params),
                 affine=affine_model,
                 psi_mniw=np.eye(out_dim),
                 nu_mniw=2 * out_dim + 1)

prior = distributions.NormalInverseWishartMatrixNormalInverseWishart(**hypparams)
model = distributions.BayesianLinearGaussianWithNoisyInputs(prior)
model = model.meanfield_update(data)

print("\n ML transf. x"+"\n", model.mu, "\n"+"ML covariance x"+"\n", model.sigma_niw)
print("True transf. x"+"\n", input_dist.mu, "\n"+"True covariance x"+"\n", input_dist.sigma, "\n")
print("ML transf. y"+"\n", model.A, "\n"+"ML covariance y"+"\n", model.sigma)
print("True transf. y"+"\n", output_dist.A, "\n"+"True covariance y"+"\n", output_dist.sigma)
