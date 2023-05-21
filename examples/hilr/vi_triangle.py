import os
import argparse

import numpy as np
import numpy.random as npr

import mimo

from mimo.mixtures import BayesianMixtureOfMixtureOfLinearGaussians

from mimo.distributions import Wishart
from mimo.distributions import NormalWishart

from mimo.distributions import MatrixNormalWithPrecision

from mimo.distributions import TiedGaussiansWithScaledPrecision
from mimo.distributions import TiedGaussiansWithHierarchicalNormalWisharts

from mimo.distributions import TiedAffineLinearGaussiansWithMatrixNormalWisharts

from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet

from mimo.mixtures import BayesianMixtureOfLinearGaussiansWithTiedActivation

import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate hilr with a Dirichlet prior')
    parser.add_argument('--cluster_alpha', help='cluster concentration', default=2.0, type=float)
    parser.add_argument('--mixture_alpha', help='mixture concentration', default=4.0, type=float)
    parser.add_argument('--cluster_size', help='max number of models', default=4, type=int)
    parser.add_argument('--mixture_size', help='max number of models', default=8, type=int)
    parser.add_argument('--prediction', help='prediction to mode or average', default='mode')
    parser.add_argument('--early_stop', help='stopping criterion for VI', default=1e-12, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # create data
    nb_train = 650

    from scipy import signal

    input = np.linspace(-4., 4., nb_train).reshape(nb_train, 1)
    noise = 0.1 * npr.randn(nb_train).reshape(nb_train, 1)
    output = 2. * signal.sawtooth(2. * np.pi * 0.5 * input, 0.5) + noise
    # plt.scatter(input, output, s=0.75, color='k')
    # plt.show()

    # model defintion
    cluster_size = args.cluster_size
    mixture_size = args.mixture_size

    input_dim = input.shape[-1]
    output_dim = output.shape[-1]

    row_dim = output_dim
    column_dim = input_dim

    # # define gating
    gating_prior = Dirichlet(dim=cluster_size, alphas=args.cluster_alpha * np.ones((cluster_size,)))
    gating = CategoricalWithDirichlet(dim=cluster_size, prior=gating_prior)

    components = []
    for _ in range(cluster_size):
        # lower gating
        _local_gating_prior = Dirichlet(dim=mixture_size, alphas=args.mixture_alpha * np.ones((mixture_size,)))
        _local_gating = CategoricalWithDirichlet(dim=mixture_size, prior=_local_gating_prior)

        # lower components
        _local_basis_hyper_prior = NormalWishart(dim=input_dim,
                                                 mu=np.zeros((input_dim,)), kappa=1e-2,
                                                 psi=np.eye(input_dim), nu=input_dim + 1 + 1e-8)

        _local_basis_prior = TiedGaussiansWithScaledPrecision(size=mixture_size, dim=input_dim,
                                                              kappas=1e-2 * np.ones((mixture_size,)))

        _local_basis = TiedGaussiansWithHierarchicalNormalWisharts(size=mixture_size, dim=input_dim,
                                                                   hyper_prior=_local_basis_hyper_prior,
                                                                   prior=_local_basis_prior)

        _local_slope_prior = MatrixNormalWithPrecision(column_dim=input_dim, row_dim=output_dim,
                                                       M=np.zeros((output_dim, input_dim)),
                                                       K=1e-2 * np.eye(input_dim))

        _local_offset_prior = TiedGaussiansWithScaledPrecision(size=mixture_size, dim=output_dim,
                                                               mus=np.zeros((mixture_size, output_dim)),
                                                               kappas=1e-2 * np.ones((mixture_size,)))

        _local_precision_prior = Wishart(dim=output_dim,
                                         psi=np.eye(output_dim),
                                         nu=(output_dim + 1) + 1e-8)

        _local_models = TiedAffineLinearGaussiansWithMatrixNormalWisharts(size=mixture_size,
                                                                          column_dim=input_dim, row_dim=output_dim,
                                                                          slope_prior=_local_slope_prior,
                                                                          offset_prior=_local_offset_prior,
                                                                          precision_prior=_local_precision_prior)

        _mixture = BayesianMixtureOfLinearGaussiansWithTiedActivation(size=mixture_size, input_dim=input_dim,
                                                                      output_dim=output_dim, gating=_local_gating,
                                                                      basis=_local_basis, models=_local_models)

        components.append(_mixture)

    hilr = BayesianMixtureOfMixtureOfLinearGaussians(cluster_size, mixture_size,
                                                     input_dim, output_dim,
                                                     gating=gating, components=components)

    # hilr.init_transform(input, output)

    hilr.resample(input, output,
                  maxiter=10,
                  maxsubiter=10,
                  maxsubsubiter=5,
                  progress_bar=args.verbose)

    hilr.meanfield_coordinate_descent(input, output,
                                      randomize=False,
                                      maxiter=10,
                                      maxsubiter=10,
                                      maxsubsubiter=5,
                                      tol=args.early_stop,
                                      progress_bar=args.verbose)

    # predict
    mu, var, std = hilr.meanfield_prediction(input, prediction=args.prediction)

    fig, axes = plt.subplots(2, 1)

    # plot prediction
    sorter = np.argsort(input, axis=0).flatten()
    sorted_input, sorted_output = input[sorter, 0], output[sorter, 0]
    sorted_mu, sorted_std = mu[sorter, 0], std[sorter, 0]

    axes[0].scatter(sorted_input, sorted_output, s=0.75, color='k')
    axes[0].plot(sorted_input, sorted_mu, color='crimson')
    for c in [1., 2.]:
        axes[0].fill_between(sorted_input,
                             sorted_mu - c * sorted_std,
                             sorted_mu + c * sorted_std,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1))

    axes[0].set_ylabel('y')

    # plot gaussian activations
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p(x)')

    activations = hilr.meanfield_predictive_activation(sorted_input)
    activations = activations.sum(axis=1)
    plt.plot(sorted_input, activations.T)

    plt.show()
