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

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate ilr with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=10, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=50, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving VI iterations', default=3, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=1, type=int)
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=50, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--nb_train', help='size of train dataset', default=1000, type=int)
    parser.add_argument('--nb_splits', help='number of dataset splits', default=25, type=int)
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # create Chrip data
    from scipy.signal import chirp

    nb_train = args.nb_train

    x = np.linspace(0, 5, nb_train)[:, None]
    y = chirp(x, f0=2.5, f1=1., t1=2.5, method='hyperbolic') + 0.25 * npr.randn(nb_train, 1)
    data = np.hstack((x, y))

    input, output = data[:, :1], data[:, 1:]

    # prepare model
    nb_models = args.nb_models
    input_dim, output_dim = 1, 1

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
    psis = np.stack(nb_models * [3e0 * np.eye(output_dim)])
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

    import copy
    from sklearn.utils import shuffle

    # init animation
    anim = []

    # init prediction
    mu, var, std = ilr.meanfield_prediction(x=input, prediction=args.prediction)

    # plot prediction
    fig = plt.figure(figsize=(1200. / 96., 675. / 96.), dpi=96)
    plt.scatter(input, output, s=0.75, color='k')
    plt.plot(input, mu, color='crimson')

    for c in [1., 2.]:
        plt.fill_between(input.flatten(),
                         mu.flatten() - c * std.flatten(),
                         mu.flatten() + c * std.flatten(),
                         color=(0, 0, 1, 0.05))

    plt.ylim((-2.5, 2.5))
    plt.tight_layout()

    # add extra frames at beginning
    for _ in range(4):
        anim.append(fig)

    # plt.show()
    # plt.pause(1)

    # set working directory
    dataset = 'chirp'
    try:
        os.chdir(args.evalpath + '/' + dataset)
    except FileNotFoundError:
        os.makedirs(args.evalpath + '/' + dataset, exist_ok=True)
        os.chdir(args.evalpath + '/' + dataset)

    # save tikz and pdf
    import tikzplotlib

    tikzplotlib.save(dataset + '_' + str(0) + '.tex')
    plt.savefig(dataset + '_' + str(0) + '.pdf')

    split_size = int(nb_train / args.nb_splits)

    mse = np.zeros((args.nb_splits, ))
    smse = np.zeros((args.nb_splits, ))
    nb_models = np.zeros((args.nb_splits, ), dtype=np.int64)

    for n in range(args.nb_splits):
        print('Processing data split ' + str(n + 1) + ' out of ' + str(args.nb_splits))

        train_input = input[n * split_size: (n + 1) * split_size, :]
        train_output = output[n * split_size: (n + 1) * split_size, :]

        ilr.gating.prior = copy.deepcopy(ilr.gating.posterior)
        ilr.basis.prior = copy.deepcopy(ilr.basis.posterior)
        ilr.models.prior = copy.deepcopy(ilr.models.posterior)

        ilr.resample(train_input, train_output,
                     init_labels='random',
                     maxiter=args.gibbs_iters)

        for _ in range(args.super_iters):
            vlb = ilr.meanfield_coordinate_descent(train_input, train_output,
                                                   randomize=False,
                                                   tol=args.earlystop,
                                                   maxiter=args.meanfield_iters,
                                                   progressbar=args.verbose)
            # print(vlb[-1])

            ilr.gating.prior = ilr.gating.posterior
            ilr.basis.prior = ilr.basis.posterior
            ilr.models.prior = ilr.models.posterior

        # predict on all data
        mu, var, std = ilr.meanfield_prediction(x=input, prediction=args.prediction)

        # plot prediction
        fig = plt.figure(figsize=(1200. / 96., 675. / 96.), dpi=96)
        plt.scatter(input, output, s=0.75, color='k')
        plt.axvspan(train_input.min(), train_input.max(), facecolor='grey', alpha=0.1)
        plt.plot(input, mu, color='crimson')

        for c in [1., 2.]:
            plt.fill_between(input.flatten(),
                             mu.flatten() - c * std.flatten(),
                             mu.flatten() + c * std.flatten(),
                             color=(0, 0, 1, 0.05))

        plt.ylim((-2.5, 2.5))
        plt.tight_layout()

        anim.append(fig)

        # plt.show()
        # plt.pause(1)

        # set working directory
        dataset = 'chirp'
        try:
            os.chdir(args.evalpath + '/' + dataset)
        except FileNotFoundError:
            os.makedirs(args.evalpath + '/' + dataset, exist_ok=True)
            os.chdir(args.evalpath + '/' + dataset)

        # save tikz and pdf
        import tikzplotlib

        tikzplotlib.save(dataset + '_' + str(n + 1) + '.tex')
        plt.savefig(dataset + '_' + str(n + 1) + '.pdf')

    # append with extra frames
    for _ in range(3):
        anim.append(anim[-1])

    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage

    fps = 8

    def make_frame(t):
        idx = int(t * fps)
        return mplfig_to_npimage(anim[idx])

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'chirp'
    path = os.path.join(str(dataset) + '/')

    animation = VideoClip(make_frame, duration=4.0)
    animation.write_gif(path + dataset + '.gif', fps=fps)
