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
    parser.add_argument('--mixture_alpha', help='mixture concentration', default=0.1, type=float)
    parser.add_argument('--cluster_size', help='max number of models', default=50, type=int)
    parser.add_argument('--mixture_size', help='max number of models', default=5, type=int)
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--prediction', help='prediction to mode or average', default='average')
    parser.add_argument('--early_stop', help='stopping criterion for VI', default=1e-12, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    # np.random.seed(args.seed)

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
    cluster_size = args.cluster_size
    mixture_size = args.mixture_size

    input_dim = input.shape[-1]
    output_dim = output.shape[-1]

    row_dim = output_dim
    column_dim = input_dim + 1

    # define gating
    # gating_prior = Dirichlet(dim=cluster_size, alphas=args.cluster_alpha * np.ones((cluster_size,)))
    # gating = CategoricalWithDirichlet(dim=cluster_size, prior=gating_prior)

    gating_prior = TruncatedStickBreaking(dim=cluster_size, gammas=np.ones((cluster_size, )),
                                          deltas=args.cluster_alpha * np.ones((cluster_size,)))
    gating = CategoricalWithStickBreaking(dim=cluster_size, prior=gating_prior)

    components = []
    for _ in range(cluster_size):
        # lower gating
        _local_gating_prior = Dirichlet(dim=mixture_size, alphas=args.mixture_alpha * np.ones((mixture_size,)))
        _local_gating = CategoricalWithDirichlet(dim=mixture_size, prior=_local_gating_prior)

        # _local_gating_prior = TruncatedStickBreaking(dim=mixture_size, gammas=np.ones((mixture_size,)),
        #                                              deltas=args.mixture_alpha * np.ones((mixture_size,)))
        # _local_gating = CategoricalWithStickBreaking(dim=mixture_size, prior=_local_gating_prior)

        # lower components
        _local_basis_hyper_prior = NormalWishart(dim=input_dim,
                                                 mu=np.zeros((input_dim,)), kappa=1e-2,
                                                 psi=np.eye(input_dim), nu=input_dim + 1 + 1e-8)

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

    # Gibbs sampling
    hilr.resample(input, output,
                  maxiter=25,
                  maxsubiter=1,
                  maxsubsubiter=1,
                  progress_bar=args.verbose)

    # for _ in range(args.super_iters):
    #     # Meanfield VI
    #     hilr.meanfield_coordinate_descent(input, output,
    #                                       randomize=False,
    #                                       maxiter=100,
    #                                       maxsubiter=50,
    #                                       maxsubsubiter=25,
    #                                       tol=args.early_stop,
    #                                       progress_bar=args.verbose)
    #
    #     hilr.gating.prior = hilr.gating.posterior
    #     for m in range(args.cluster_size):
    #         hilr.components[m].gating.prior = hilr.components[m].gating.posterior
    #         hilr.components[m].basis.hyper_prior = hilr.components[m].basis.hyper_posterior
    #         hilr.components[m].models.slope_prior = hilr.components[m].models.slope_posterior
    #         hilr.components[m].models.offset_prior = hilr.components[m].models.offset_posterior
    #         hilr.components[m].models.precision_prior = hilr.components[m].models.precision_posterior
    #
    # # predict
    # mu, var, std = hilr.meanfield_prediction(input, prediction='average')
    #
    # # plot prediction
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_trisurf(input[:, 0], input[:, 1], mu[:, 0], cmap='plasma')
    # plt.show()
    #
    # plt.figure()
    # ax = plt.gca()
    #
    # from matplotlib import cm
    # cmap = cm.get_cmap('Paired')
    # label_colors = dict((idx, cmap(v)) for idx, v in
    #                     enumerate(np.linspace(0, 1, hilr.cluster_size, endpoint=True)))
    #
    # for k in range(hilr.cluster_size):
    #     if hilr.gating.posterior.mean()[k] > 1e-2:
    #         mus = hilr.components[k].basis.posterior.mus
    #         precs = hilr.components[k].basis.posterior.lmbdas
    #
    #         for m in range(hilr.components[k].size):
    #             t = np.hstack([np.arange(0, 2 * np.pi, 0.01), 0])
    #             circle = np.vstack([np.sin(t), np.cos(t)])
    #             ellipse = np.dot(np.linalg.cholesky(np.linalg.inv(precs[m])), circle)
    #
    #             point = ax.scatter(mus[m, 0], mus[m, 1], marker='D', s=4, c=label_colors[k])
    #             line, = ax.plot(ellipse[0, :] + mus[m, 0], ellipse[1, :] + mus[m, 1],
    #                             linestyle='-', linewidth=2, c=label_colors[k])
    #
    # ax.set_xlim(-6.0, 6.0)
    # ax.set_ylim(-6.0, 6.0)
    #
    # plt.show()
