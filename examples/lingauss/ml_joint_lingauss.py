import numpy as np
import numpy.random as npr

from mimo import distributions

npr.seed(1337)

in_dim = 1
out_dim = 1

_A = 1. * npr.randn(out_dim, in_dim)

nb_samples = 4000

input_dist = distributions.Gaussian(mu=npr.randn(in_dim), sigma=5. * np.diag(npr.rand(in_dim)))
x = input_dist.rvs(size=nb_samples)
x = np.asarray(x)

output_dist = distributions.LinearGaussian(A=_A, sigma=25e-2 * np.eye(out_dim), affine=False)
data = output_dist.rvs(size=nb_samples, x=x)

affine = False
n_params = in_dim + 1 if affine else in_dim

hypparams = dict(mu=np.zeros((in_dim,)),
                 kappa=0.05,
                 psi_niw=np.eye(in_dim),
                 nu_niw=in_dim + 1,
                 M=np.zeros((out_dim, n_params)),
                 V=1. * np.eye(n_params),
                 affine=affine,
                 psi_mniw=np.eye(out_dim),
                 nu_mniw=out_dim + 1)

prior = distributions.NormalInverseWishartMatrixNormalInverseWishart(**hypparams)
model = distributions.BayesianJointLinearGaussian(prior)
model = model.max_likelihood(data)

print("True transf. x"+"\n", input_dist.mu, "\n"+"True covariance x"+"\n", input_dist.sigma)
print("MF transf. x"+"\n", model.gaussian.mu, "\n"+"MF covariance x"+"\n", model.gaussian.sigma)
print("------------------------------------------------------------------------------------------------")
print("True transf. y"+"\n", output_dist.A, "\n"+"True covariance y"+"\n", output_dist.sigma)
print("MF transf. y"+"\n", model.linear_gaussian.A, "\n"+"MF covariance y"+"\n", model.linear_gaussian.sigma)
