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

from mimo.distributions import TruncatedStickBreaking
from mimo.distributions import CategoricalWithStickBreaking

from mimo.mixtures import BayesianMixtureOfLinearGaussiansWithTiedActivation

import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate ilr with a Stick-breaking prior')
    parser.add_argument('--data_path', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--eval_path', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--cluster_alpha', help='cluster concentration', default=5., type=float)
    parser.add_argument('--mixture_alpha', help='mixture concentration', default=1., type=float)
    parser.add_argument('--cluster_size', help='max number of models', default=25, type=int)
    parser.add_argument('--mixture_size', help='max number of models', default=5, type=int)
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=0, type=int)
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=250, type=int)
    parser.add_argument('--prediction', help='prediction to mode or average', default='average')
    parser.add_argument('--early_stop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # create Sine data
    nb_samples = 2000

    true_output = np.zeros((nb_samples,))
    true_input = np.zeros((nb_samples,))

    data = np.zeros((nb_samples, 2))
    step = 10. * np.pi / nb_samples
    for i in range(data.shape[0]):
        true_input[i] = i * step
        data[i, 0] = true_input[i] + 0.1 * npr.randn()
        true_output[i] = 3 * np.sin(true_input[i])
        data[i, 1] = true_output[i] + 0.3 * npr.randn()

    from itertools import chain
    r = list(chain(range(0, 500), range(1000, 1500)))
    train_data = data[r, :]

    # training data
    input, output = data[:, :1], data[:, 1:]
    train_input, train_output = train_data[:, :1], train_data[:, 1:]

    # model defintion
    cluster_size = args.cluster_size
    mixture_size = args.mixture_size

    input_dim = input.shape[-1]
    output_dim = output.shape[-1]

    row_dim = output_dim
    column_dim = input_dim

    # define gating
    gating_prior = TruncatedStickBreaking(dim=cluster_size, gammas=np.ones((cluster_size, )),
                                          deltas=args.cluster_alpha * np.ones((cluster_size,)))
    gating = CategoricalWithStickBreaking(dim=cluster_size, prior=gating_prior)

    components = []
    for _ in range(cluster_size):
        # lower gating
        _local_gating_prior = TruncatedStickBreaking(dim=mixture_size, gammas=np.ones((mixture_size,)),
                                                     deltas=args.mixture_alpha * np.ones((mixture_size,)))
        _local_gating = CategoricalWithStickBreaking(dim=mixture_size, prior=_local_gating_prior)

        # lower components
        _local_basis_hyper_prior = NormalWishart(dim=input_dim,
                                                 mu=np.zeros((input_dim,)), kappa=1e-2,
                                                 psi=1e2 * np.eye(input_dim), nu=input_dim + 1 + 1e-8)

        _local_basis_prior = TiedGaussiansWithScaledPrecision(size=mixture_size, dim=input_dim,
                                                              kappas=np.ones((mixture_size,)))

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
                                         psi=1e1 * np.eye(output_dim),
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

    hilr.init_transform(train_input, train_output)

    # Gibbs sampling
    hilr.resample(train_input, train_output,
                  maxiter=50,
                  maxsubiter=25,
                  maxsubsubiter=10,
                  progress_bar=args.verbose)

    for _ in range(args.super_iters):
        # Meanfield VI
        hilr.meanfield_coordinate_descent(train_input, train_output,
                                          randomize=False,
                                          maxiter=25,
                                          maxsubiter=25,
                                          maxsubsubiter=10,
                                          tol=args.early_stop,
                                          progress_bar=args.verbose)

        hilr.gating.prior = hilr.gating.posterior
        for m in range(args.cluster_size):
            hilr.components[m].gating.prior = hilr.components[m].gating.posterior
            hilr.components[m].basis.hyper_prior = hilr.components[m].basis.hyper_posterior
            hilr.components[m].models.slope_prior = hilr.components[m].models.slope_posterior
            hilr.components[m].models.offset_prior = hilr.components[m].models.offset_posterior
            hilr.components[m].models.precision_prior = hilr.components[m].models.precision_posterior

    # predict
    mu, var, std = hilr.meanfield_prediction(input, prediction=args.prediction)

    fig, axes = plt.subplots(2, 1)

    # plot prediction
    sorter = np.argsort(input, axis=0).flatten()
    sorted_input, sorted_output = input[sorter, 0], output[sorter, 0]
    sorted_mu, sorted_std = mu[sorter, 0], std[sorter, 0]

    axes[0].plot(true_input, true_output, '--k')
    axes[0].scatter(train_input, train_output, marker='+', s=1.25, color='k')
    axes[0].plot(sorted_input, sorted_mu, color='crimson')
    for c in [1., 2., 3.]:
        axes[0].fill_between(sorted_input,
                             sorted_mu - c * sorted_std,
                             sorted_mu + c * sorted_std,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1))

    axes[0].set_ylabel('y')
    axes[0].set_ylim(-10.0, 10.0)

    # plot gaussian activations
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p(x)')

    activations = hilr.meanfield_predictive_activation(sorted_input)
    activations = activations.sum(axis=1)
    plt.plot(sorted_input, activations.T)

    # # set working directory
    # dataset = 'step'
    # try:
    #     os.chdir(args.eval_path + '/' + dataset)
    # except FileNotFoundError:
    #     os.makedirs(args.eval_path + '/' + dataset, exist_ok=True)
    #     os.chdir(args.eval_path + '/' + dataset)
    #
    # # save tikz and pdf
    # import tikzplotlib
    #
    # tikzplotlib.save(dataset + '.tex')
    # plt.savefig(dataset + '.pdf')

    plt.show()
