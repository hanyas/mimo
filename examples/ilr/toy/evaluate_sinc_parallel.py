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

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


def create_job(kwargs):
    args = kwargs.pop('arguments')
    seed = kwargs.pop('seed')

    input = kwargs.pop('train_input')
    output = kwargs.pop('train_output')

    # set random seed
    np.random.seed(seed)

    # model defintion
    nb_models = args.nb_models
    input_dim = input.shape[-1]
    output_dim = output.shape[-1]

    row_dim = output_dim
    column_dim = input_dim + 1 if args.affine else input_dim

    # initialize Normal
    mus = np.zeros((nb_models, input_dim))
    kappas = 1e-2 * np.ones((nb_models,))
    psis = np.stack(nb_models * [1e2 * np.eye(input_dim)])
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
    psis = np.stack(nb_models * [1e0 * np.eye(output_dim)])
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

    ilr = BayesianMixtureOfLinearGaussians(gating=gating, basis=basis, models=models)

    ilr.init_transform(input, output)

    for _ in range(args.super_iters):
        # Gibbs sampling
        ilr.resample(input, output,
                     labels='random',
                     maxiter=args.gibbs_iters,
                     progressbar=args.verbose)

        if args.stochastic:
            # Stochastic meanfield VI
            ilr.meanfield_stochastic_descent(input, output,
                                             maxiter=args.svi_iters,
                                             stepsize=args.svi_stepsize,
                                             batchsize=args.svi_batchsize)
        if args.deterministic:
            # Meanfield VI
            ilr.meanfield_coordinate_descent(input, output,
                                             randomize=False,
                                             maxiter=args.meanfield_iters,
                                             tol=args.earlystop,
                                             progressbar=args.verbose)

        ilr.gating.prior = ilr.gating.posterior
        ilr.basis.prior = ilr.basis.posterior
        ilr.models.prior = ilr.models.posterior

    return ilr


def parallel_ilr_inference(nb_jobs=50, **kwargs):
    kwargs_list = []
    for n in range(nb_jobs):
        _kwargs = {'seed': n,
                   'train_input': kwargs['train_input'][n],
                   'train_output': kwargs['train_output'][n],
                   'arguments': kwargs['arguments']}
        kwargs_list.append(_kwargs)

    ilrs = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=1, backend='loky')\
        (map(delayed(create_job), kwargs_list))

    return ilrs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate ilr with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=24, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=100, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=100, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=2, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=0, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=250, type=int)
    parser.add_argument('--svi_iters', help='SVI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=512, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-4, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=False)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # sample dataset
    nb_samples = 2500
    input = np.linspace(-10., 10., nb_samples).reshape(nb_samples, 1)
    noise = lambda x: 0.05 + 0.2 * (1. + np.sin(2. * x)) / (1. + np.exp(-0.2 * x))
    target = np.sinc(input) + noise(input) * np.random.randn(len(input), 1)
    mean = np.sinc(input)

    data = np.hstack((input, target))

    # shuffle data
    from sklearn.utils import shuffle

    data = shuffle(data)

    # split to nb_seeds train datasets
    from sklearn.model_selection import ShuffleSplit

    spliter = ShuffleSplit(n_splits=args.nb_seeds, test_size=0.2)

    train_inputs, train_outputs = [], []
    for train_index, _ in spliter.split(data):
        train_inputs.append(data[train_index, :1])
        train_outputs.append(data[train_index, 1:])

    # train
    ilrs = parallel_ilr_inference(nb_jobs=args.nb_seeds,
                                  train_input=train_inputs,
                                  train_output=train_outputs,
                                  arguments=args)

    # Evaluation over multiple seeds to get confidence
    mu, std, = [], []
    for ilr in ilrs:
        _mu, _var, _std = ilr.meanfield_prediction(input, prediction=args.prediction)
        mu.append(_mu)
        std.append(_std)

    mu = np.asarray(mu).squeeze()
    std = np.asarray(std).squeeze()

    # calcule means and confidence intervals
    mu_avg, mu_std = np.mean(mu, axis=0), np.std(mu, axis=0)
    std_avg, std_std = np.mean(std, axis=0), np.std(std, axis=0)

    # plot mean and standard deviation of mean estimation
    w, h = plt.figaspect(0.67)
    fig, axes = plt.subplots(2, 1, figsize=(w, h))

    # plot data and prediction
    axes[0].plot(input, mean, 'k--', zorder=10)
    axes[0].scatter(input, target, s=0.75, facecolors='none', edgecolors='grey', zorder=1)
    axes[0].plot(input, mu_avg, '-r', zorder=5)
    for c in [1., 2.]:
        axes[0].fill_between(input.flatten(),
                             mu_avg - c * mu_std,
                             mu_avg + c * mu_std,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1), zorder=1)

    # plot mean and standard deviation of data generation / estimated noise level
    axes[1].plot(input, noise(input), 'k--', zorder=10)
    axes[1].plot(input, std_avg, '-r', zorder=5)
    for c in [1., 2.]:
        axes[1].fill_between(input.flatten(),
                             std_avg - c * std_std,
                             std_avg + c * std_std,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1), zorder=1)
    plt.tight_layout()

    # # set working directory
    # dataset = 'sinc'
    # try:
    #     os.chdir(args.evalpath + '/' + dataset)
    # except FileNotFoundError:
    #     os.makedirs(args.evalpath + '/' + dataset, exist_ok=True)
    #     os.chdir(args.evalpath + '/' + dataset)
    #
    # # save figs
    # import tikzplotlib
    #
    # tikzplotlib.save(dataset + '_mean.tex')
    # plt.savefig(dataset + '_mean.pdf')

    plt.show()

    # show on learned example
    choice = np.random.choice(args.nb_seeds, 1)[0]
    w, h = plt.figaspect(0.67)
    fig, axes = plt.subplots(2, 1, figsize=(w, h))

    # plot data and prediction
    axes[0].plot(input, mean, 'k--', zorder=10)
    axes[0].plot(input, mean + 2 * noise(input), 'g--', zorder=10)
    axes[0].plot(input, mean - 2 * noise(input), 'g--', zorder=10)
    axes[0].scatter(input, target, s=0.75, facecolors='none', edgecolors='grey', zorder=1)

    axes[0].plot(input, mu[choice], '-r', zorder=5)
    axes[0].plot(input, mu[choice] + 2 * std[choice], '-b', zorder=5)
    axes[0].plot(input, mu[choice] - 2 * std[choice], '-b', zorder=5)

    # plot gaussian activations
    activations = ilrs[choice].meanfield_predictive_activation(input)
    axes[1].plot(input, activations.T)

    # # save figs
    # import tikzplotlib
    #
    # tikzplotlib.save(dataset + '_example.tex')
    # plt.savefig(dataset + '_example.pdf')

    plt.show()
