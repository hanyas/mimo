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
    parser.add_argument('--nb_models', help='max number of models', default=100, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=2, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=250, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=True)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=False)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=250, type=int)
    parser.add_argument('--svi_iters', help='SVI iterations', default=250, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=25e-2, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=256, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--early_stop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    npr.seed(args.seed)

    # def cross_2d(x, y):
    #     return np.maximum(np.exp(-10.0 * x * x), np.exp(-50.0 * y * y),
    #                       1.25 * np.exp(-5.0 * (x * x + y * y)))

    # x = npr.uniform(-1., 1., size=(nb_samples, 1))
    # y = npr.uniform(-1., 1., size=(nb_samples, 1))

    def sine_sqrt(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))

    X = np.arange(-6., 6., 0.12)
    Y = np.arange(-6., 6., 0.12)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            Z[i, j] = sine_sqrt(np.array([X[i, j]]), np.array([Y[i, j]]))

    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='plasma', edgecolor='none')
    plt.show()

    # training data
    input = np.vstack([X.ravel(), Y.ravel()]).T
    output = np.ravel(Z)[:, None] + 0.01 * npr.randn(100 * 100, 1)

    # model defintion
    nb_models = args.nb_models
    input_dim = input.shape[-1]
    output_dim = output.shape[-1]

    row_dim = output_dim
    column_dim = input_dim + 1 if args.affine else input_dim

    # initialize Normal
    mus = np.zeros((nb_models, input_dim))
    kappas = 1e-2 * np.ones((nb_models,))
    psis = np.stack(nb_models * [1e1 * np.eye(input_dim)])
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
    psis = np.stack(nb_models * [1e1 * np.eye(output_dim)])
    nus = (output_dim + 1) * np.ones((nb_models,)) + 1e-16

    models_prior = TiedMatrixNormalWisharts(nb_models, column_dim, row_dim,
                                            Ms=Ms, Ks=Ks, psis=psis, nus=nus)

    models = TiedLinearGaussiansWithMatrixNormalWisharts(nb_models, column_dim, row_dim,
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
                                             stepsize=args.svi_stepsize,
                                             batchsize=args.svi_batchsize)
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

    # predict on training
    mu, var, std = ilr.meanfield_prediction(input, prediction=args.prediction)

    # plot prediction
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(input[:, 0], input[:, 1], mu[:, 0], cmap='plasma')
    plt.show()

    plt.figure()
    ax = plt.gca()
    mus, precs = ilr.basis.posterior.mean()

    for k in range(ilr.size):
        if ilr.gating.posterior.mean()[k] > 1e-2:
            t = np.hstack([np.arange(0, 2 * np.pi, 0.01), 0])
            circle = np.vstack([np.sin(t), np.cos(t)])
            ellipse = np.dot(np.linalg.cholesky(np.linalg.inv(precs[k])), circle)

            point = ax.scatter(mus[k, 0], mus[k, 1], marker='D', s=4)
            line, = ax.plot(ellipse[0, :] + mus[k, 0], ellipse[1, :] + mus[k, 1],
                            linestyle='-', linewidth=2)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)

    plt.show()
