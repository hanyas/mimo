import os
import argparse

import numpy as np
import numpy.random as npr

import mimo

from mimo.distributions import StackedNormalWisharts
from mimo.distributions import StackedMatrixNormalWisharts
from mimo.distributions import TiedMatrixNormalWisharts

from mimo.distributions import StackedGaussiansWithNormalWisharts
from mimo.distributions import StackedLinearGaussiansWithMatrixNormalWisharts
from mimo.distributions import TiedLinearGaussiansWithMatrixNormalWisharts

from mimo.distributions import TruncatedStickBreaking
from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet
from mimo.distributions import CategoricalWithStickBreaking

from mimo.mixtures import BayesianMixtureOfLinearGaussians

import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate ilr with a Stick-breaking prior')
    parser.add_argument('--data_path', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--eval_path', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=10, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=10, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=2, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=0, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=True)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=False)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=250, type=int)
    parser.add_argument('--svi_iters', help='SVI iterations', default=500, type=int)
    parser.add_argument('--svi_step_size', help='SVI step size', default=5e-1, type=float)
    parser.add_argument('--svi_batch_size', help='SVI batch size', default=32, type=int)
    parser.add_argument('--prediction', help='prediction to mode or average', default='mode')
    parser.add_argument('--early_stop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    # np.random.seed(args.seed)

    # create data
    nb_train = 160

    input, mean = [], []

    input.append(np.linspace(-2., -1, int(nb_train / 4)).reshape(int(nb_train / 4), 1))
    input.append(np.linspace(-1., 0, int(nb_train / 4)).reshape(int(nb_train / 4), 1))
    input.append(np.linspace(0, 1., int(nb_train / 4)).reshape(int(nb_train / 4), 1))
    input.append(np.linspace(1, 2., int(nb_train / 4)).reshape(int(nb_train / 4), 1))

    mean.append(np.ones((int(nb_train / 4), 1)) * 1.)
    mean.append(np.ones((int(nb_train / 4), 1)) * 3.)
    mean.append(np.ones((int(nb_train / 4), 1)) * 0.)
    mean.append(np.ones((int(nb_train / 4), 1)) * 4.)

    input, mean = np.vstack(input), np.vstack(mean)
    noise = 0.1 * npr.randn(nb_train).reshape(nb_train, 1)
    output = mean + noise

    # model defintion
    nb_models = args.nb_models
    input_dim = input.shape[-1]
    output_dim = output.shape[-1]

    row_dim = output_dim
    column_dim = input_dim + 1 if args.affine else input_dim

    # initialize Normal
    mus = np.zeros((nb_models, input_dim))
    kappas = 1e-2 * np.ones((nb_models,))
    psis = np.stack(nb_models * [1e0 * np.eye(input_dim)])
    nus = (input_dim + 1) * np.ones((nb_models,)) + 1e-16

    basis_prior = StackedNormalWisharts(size=nb_models, dim=input_dim,
                                        mus=mus, kappas=kappas,
                                        psis=psis, nus=nus)

    basis = StackedGaussiansWithNormalWisharts(size=nb_models,
                                               dim=input_dim,
                                               prior=basis_prior)

    # initialize Matrix-Normal
    Ms = np.zeros((nb_models, row_dim, column_dim))
    Ks = np.stack(nb_models * [1e-2 * np.eye(column_dim)])
    psis = np.stack(nb_models * [1e2 * np.eye(output_dim)])
    nus = (output_dim + 1) * np.ones((nb_models,)) + 1e-16

    models_prior = StackedMatrixNormalWisharts(nb_models, column_dim, row_dim,
                                               Ms=Ms, Ks=Ks, psis=psis, nus=nus)

    models = StackedLinearGaussiansWithMatrixNormalWisharts(nb_models, column_dim, row_dim,
                                                            models_prior, affine=args.affine)

    # define gating
    if args.prior == 'stick-breaking':
        gammas = np.ones((args.nb_models,))
        betas = np.ones((args.nb_models,)) * args.alpha
        gating_prior = TruncatedStickBreaking(nb_models, gammas, betas)
        gating = CategoricalWithStickBreaking(nb_models, gating_prior)
    else:
        alphas = np.ones((args.nb_models,)) * args.alpha
        gating_prior = Dirichlet(nb_models, alphas)
        gating = CategoricalWithDirichlet(nb_models, gating_prior)

    ilr = BayesianMixtureOfLinearGaussians(size=nb_models,
                                           input_dim=input_dim, output_dim=output_dim,
                                           gating=gating, basis=basis, models=models)

    ilr.init_transform(input, output)

    # Gibbs sampling
    ilr.resample(input, output,
                 init_labels='random',
                 maxiter=args.gibbs_iters,
                 progress_bar=args.verbose)

    for _ in range(args.super_iters):
        if args.stochastic:
            # Stochastic meanfield VI
            ilr.meanfield_stochastic_descent(input, output,
                                             randomize=False,
                                             maxiter=args.svi_iters,
                                             step_size=args.svi_step_size,
                                             batch_size=args.svi_batch_size)
        if args.deterministic:
            # Meanfield VI
            ilr.meanfield_coordinate_descent(input, output,
                                             randomize=False,
                                             maxiter=args.meanfield_iters,
                                             tol=args.early_stop,
                                             progress_bar=args.verbose)

        # ilr.gating.prior = ilr.gating.posterior
        ilr.basis.prior = ilr.basis.posterior
        ilr.models.prior = ilr.models.posterior

    # predict
    mu, var, std = ilr.meanfield_prediction(input, prediction=args.prediction)

    fig, axes = plt.subplots(2, 1)

    # plot prediction
    sorter = np.argsort(input, axis=0).flatten()
    sorted_input, sorted_output = input[sorter, 0], output[sorter, 0]
    sorted_mu, sorted_std = mu[sorter, 0], std[sorter, 0]

    axes[0].scatter(sorted_input, sorted_output, s=0.75, color='k')
    axes[0].plot(sorted_input, sorted_mu, color='crimson')
    for c in [1., 2., 3.]:
        axes[0].fill_between(sorted_input,
                             sorted_mu - c * sorted_std,
                             sorted_mu + c * sorted_std,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1))

    axes[0].set_ylabel('y')

    # plot gaussian activations
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p(x)')

    activations = ilr.meanfield_predictive_activation(sorted_input)
    axes[1].plot(sorted_input, activations.T)

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
