import numpy as np
import numpy.random as npr

from mimo import distributions

in_dim = 2
out_dim = 1

_A = 1. * npr.randn(out_dim, in_dim)

nb_samples = 4000


dist1 = distributions.Gaussian(mu=npr.randn(in_dim), sigma=5. * np.diag(npr.rand(in_dim)))
x = dist1.rvs(size=nb_samples)
x = np.asarray(x)

dist2 = distributions.LinearGaussian(A=_A, sigma=2.5e-1 * np.eye(out_dim), affine=False)

data = dist2.rvs(size=nb_samples, x=x)

affine_model = False
if affine_model:
    in_dim_mniw = in_dim + 1
else:
    in_dim_mniw = in_dim

hypparams = dict(mu=np.zeros((in_dim,)),
                 kappa=0.05,
                 psi_niw=np.eye(in_dim),
                 nu_niw=2 * in_dim + 1,
                 M=np.zeros((out_dim, in_dim_mniw)),
                 V=1. * np.eye(in_dim_mniw),
                 affine=affine_model,
                 psi_mniw=np.eye(out_dim),
                 nu_mniw=2 * out_dim + 1)


prior = distributions.NormalInverseWishartMatrixNormalInverseWishart(**hypparams)

model = distributions.BayesianLinearGaussianWithNoisyInputs(prior)
model = model.meanfieldupdate(data)

print("\n ML transf. x"+"\n", model.mu, "\n"+"ML covariance x"+"\n", model.sigma_niw)
print("True transf. x"+"\n", dist1.mu, "\n"+"True covariance x"+"\n", dist1.sigma, "\n")
print("ML transf. y"+"\n", model.A, "\n"+"ML covariance y"+"\n", model.sigma)
print("True transf. y"+"\n", dist2.A, "\n"+"True covariance y"+"\n", dist2.sigma)