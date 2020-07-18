import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

from mimo.distributions import NormalWishart
from mimo.distributions import MatrixNormalWishart

from mimo.distributions import GaussianWithNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishart

from mimo.distributions import StickBreaking
from mimo.distributions import CategoricalWithStickBreaking

from mimo.mixtures import BayesianMixtureOfLinearGaussians

from mimo.util.text import progprint_xrange

from dataclasses import dataclass


@dataclass
class ILR(object):

    nb_models: int
    affine: bool
    gating_alpha: float
    basis_psi: float
    basis_kappa: float
    model_psi: float
    model_K: float

    super_iters: int
    gibbs_iters: int
    vi_iters: int
    vi_earlystop: float
    svi_iters: int
    svi_batch_size: int
    svi_step_size: float


class InfiniteLinearRegression:
    """
    This class serves as an abstract RegressionModel template
    """
    def __init__(self, input_dim, output_dim,
                 nb_models, affine,
                 gating_alpha,
                 basis_psi, basis_kappa,
                 model_psi, model_K,
                 super_iters, gibbs_iters,
                 vi_iters, vi_earlystop,
                 svi_iters, svi_batch_size,
                 svi_step_size):
        """
        The constructor for RegressionModel.

        Parameters:
            input_dim (int): The number of input dimensions of the regression model.
            output_dim (int): The number of output dimensions of the regression model.

        """
        self.k = input_dim
        self.d = output_dim

        self.nb_models = nb_models

        self.affine = affine

        # number of parameters per local regressor
        self.nb_params = self.k + 1 if self.affine else self.k

        self.basis_prior = []
        self.model_prior = []

        for n in range(self.nb_models):
            basis_hypparams = dict(mu=np.zeros((self.k,)),
                                   psi=np.eye(self.k) * basis_psi,
                                   kappa=basis_kappa, nu=self.k + 2)

            self.basis_prior.append(NormalWishart(**basis_hypparams))

            model_hypparams = dict(M=np.zeros((self.d, self.nb_params)),
                                   K=np.eye(self.nb_params) * model_K,
                                   psi=np.eye(self.d) * model_psi, nu=self.d + 2)

            self.model_prior.append(MatrixNormalWishart(**model_hypparams))

        gating_hypparams = dict(K=self.nb_models,
                                gammas=np.ones((self.nb_models,)),
                                deltas=np.ones((self.nb_models,)) * gating_alpha)
        self.gating_prior = StickBreaking(**gating_hypparams)

        self.regressor = None

        self.super_iters = super_iters
        self.gibbs_iters = gibbs_iters

        self.vi_iters = vi_iters
        self.vi_earlystop = vi_earlystop

        self.svi_iters = svi_iters
        self.svi_batch_size = svi_batch_size
        self.svi_step_size = svi_step_size

    def fit(self, input, output, verbose=True):
        """
        Fits the model to the training data.
        """

        self.regressor = BayesianMixtureOfLinearGaussians(gating=CategoricalWithStickBreaking(self.gating_prior),
                                                          basis=[GaussianWithNormalWishart(self.basis_prior[i]) for i in range(self.nb_models)],
                                                          models=[LinearGaussianWithMatrixNormalWishart(self.model_prior[i], affine=self.affine)
                                                                  for i in range(self.nb_models)])

        self.regressor.add_data(output, input, whiten=False)

        for i in range(self.super_iters):
            # Gibbs sampling
            if verbose:
                print("Gibbs Sampling")

            gibbs_iter = range(self.gibbs_iters) if not verbose \
                else progprint_xrange(self.gibbs_iters)

            for _ in gibbs_iter:
                self.regressor.resample()

            # Stochastic meanfield VI
            if verbose:
                print('Stochastic Variational Inference')

            svi_iter = range(self.svi_iters) if not verbose \
                else progprint_xrange(self.svi_iters)

            prob = self.svi_batch_size / float(len(input))
            for _ in svi_iter:
                minibatch = npr.permutation(len(input))[:self.svi_batch_size]
                self.regressor.meanfield_sgdstep(y=output[minibatch, :], x=input[minibatch, :],
                                                 prob=prob, stepsize=self.svi_step_size)

            # Meanfield VI
            if verbose:
                print("Variational Inference")
            self.regressor.meanfield_coordinate_descent(tol=self.vi_earlystop,
                                                        maxiter=self.vi_iters,
                                                        progprint=verbose)

            if self.super_iters > 1 and i + 1 < self.super_iters:
                self.regressor.gating.prior = self.regressor.gating.posterior
                for n in range(self.regressor.size):
                    self.regressor.basis[n].prior = self.regressor.basis[n].posterior
                    self.regressor.models[n].prior = self.regressor.models[n].posterior

    def predict(self, X):
        mu, var, _ = self.regressor.meanfield_prediction(X, prediction='average')
        return mu, var

    def elatoric(self):
        var = self.regressor.meanfield_elatoric()
        return np.mean(var, axis=-1)