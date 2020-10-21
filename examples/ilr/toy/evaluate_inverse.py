import os
import argparse

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import mimo
from mimo.distributions import NormalWishart
from mimo.distributions import MatrixNormalWishart
from mimo.distributions import GaussianWithNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishart

from mimo.distributions import TruncatedStickBreaking
from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet
from mimo.distributions import CategoricalWithStickBreaking

from mimo.mixtures import BayesianMixtureOfLinearGaussians

import matplotlib.pyplot as plt
from tqdm import tqdm

import pathos
from pathos.pools import _ProcessPool as Pool

nb_cores = pathos.multiprocessing.cpu_count()


def _job(kwargs):
    args = kwargs.pop('arguments')
    seed = kwargs.pop('seed')

    input = kwargs.pop('train_input')
    target = kwargs.pop('train_target')

    input_dim = input.shape[-1]
    target_dim = target.shape[-1]

    # set random seed
    np.random.seed(seed)

    nb_params = input_dim
    if args.affine:
        nb_params += 1

    basis_prior = []
    models_prior = []

    # initialize Normal
    psi_nw = 1e1
    kappa = 1e-2

    # initialize Matrix-Normal
    psi_mnw = 1e0
    K = 1e-3

    for n in range(args.nb_models):
        basis_hypparams = dict(mu=np.zeros((input_dim,)),
                               psi=np.eye(input_dim) * psi_nw,
                               kappa=kappa, nu=input_dim + 1)

        aux = NormalWishart(**basis_hypparams)
        basis_prior.append(aux)

        models_hypparams = dict(M=np.zeros((target_dim, nb_params)),
                                K=K * np.eye(nb_params), nu=target_dim + 1,
                                psi=np.eye(target_dim) * psi_mnw)

        aux = MatrixNormalWishart(**models_hypparams)
        models_prior.append(aux)

    # define gating
    if args.prior == 'stick-breaking':
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)),
                                deltas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = TruncatedStickBreaking(**gating_hypparams)

        ilr = BayesianMixtureOfLinearGaussians(gating=CategoricalWithStickBreaking(gating_prior),
                                               basis=[GaussianWithNormalWishart(basis_prior[i])
                                                      for i in range(args.nb_models)],
                                               models=[LinearGaussianWithMatrixNormalWishart(models_prior[i], affine=args.affine)
                                                       for i in range(args.nb_models)])

    else:
        gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = Dirichlet(**gating_hypparams)

        ilr = BayesianMixtureOfLinearGaussians(gating=CategoricalWithDirichlet(gating_prior),
                                               basis=[GaussianWithNormalWishart(basis_prior[i])
                                                      for i in range(args.nb_models)],
                                               models=[LinearGaussianWithMatrixNormalWishart(models_prior[i], affine=args.affine)
                                                       for i in range(args.nb_models)])
    ilr.add_data(target, input, whiten=False,
                 labels_from_prior=True)

    # Gibbs sampling
    ilr.resample(maxiter=args.gibbs_iters,
                 progprint=args.verbose)

    for _ in range(args.super_iters):
        if args.stochastic:
            # Stochastic meanfield VI
            ilr.meanfield_stochastic_descent(maxiter=args.svi_iters,
                                             stepsize=args.svi_stepsize,
                                             batchsize=args.svi_batchsize)
        if args.deterministic:
            # Meanfield VI
            ilr.meanfield_coordinate_descent(tol=args.earlystop,
                                             maxiter=args.meanfield_iters,
                                             progprint=args.verbose)

        ilr.gating.prior = ilr.gating.posterior
        for i in range(ilr.likelihood.size):
            ilr.basis[i].prior = ilr.basis[i].posterior
            ilr.models[i].prior = ilr.models[i].posterior

    return ilr


def parallel_ilr_inference(nb_jobs=50, **kwargs):
    kwargs_list = []
    for n in range(nb_jobs):
        kwargs['seed'] = n
        kwargs_list.append(kwargs.copy())

    with Pool(processes=min(nb_jobs, nb_cores),
              initializer=tqdm.set_lock,
              initargs=(tqdm.get_lock(),)) as p:
        res = p.map(_job, kwargs_list)

    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate ilr with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=5, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=10, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=25, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=1000, type=int)
    parser.add_argument('--svi_iters', help='SVI iterations', default=250, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=256, type=int)
    parser.add_argument('--prediction', help='prediction to mode or average', default='mode')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # create data
    noise = npr.normal(0, 1, (200, 1)) * 0.05
    target = npr.uniform(0, 1, (200, 1))
    input = target + 0.3 * np.sin(2. * np.pi * target) + noise

    ilr = parallel_ilr_inference(nb_jobs=args.nb_seeds,
                                 train_input=input,
                                 train_target=target,
                                 arguments=args)[0]

    # mean prediction
    mu, _, _ = ilr.meanfield_prediction(input, prediction='average')

    # metrics
    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

    evar = explained_variance_score(target, mu)
    mse = mean_squared_error(target, mu)
    smse = 1. - r2_score(target, mu)

    print('MEAN - EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'Components:', len(ilr.used_labels))

    # creat plot for mean vs mode prediction and gaussian activations
    fig, axes = plt.subplots(2, 1)

    axes[0].scatter(input, target, facecolors='none', edgecolors='k', linewidth=0.5)
    axes[0].scatter(input, mu, marker='x', c='b', linewidth=0.5)
    plt.ylabel('y')

    # mean prediction
    mu, _, _ = ilr.meanfield_prediction(input, prediction='mode')

    # metrics
    from sklearn.metrics import explained_variance_score, mean_squared_error

    mse = mean_squared_error(target, mu)
    evar = explained_variance_score(target, mu, multioutput='variance_weighted')
    smse = 1. - r2_score(target, mu, multioutput='variance_weighted')

    print('Mode - EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'Components:', len(ilr.used_labels))

    axes[0].scatter(input, mu, marker='D', facecolors='none', edgecolors='r', linewidth=0.5)

    # plot gaussian activations
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p(x)')

    sorted_input = np.sort(input, axis=0)
    activations = ilr.meanfield_predictive_activation(sorted_input)

    colours = ['green', 'orange', 'purple']
    for k, i in enumerate(ilr.used_labels):
        axes[1].plot(sorted_input, activations[:, i], color=colours[k])

    # set working directory
    dataset = 'inverse'
    try:
        os.chdir(args.evalpath + '/' + dataset)
    except FileNotFoundError:
        os.makedirs(args.evalpath + '/' + dataset, exist_ok=True)
        os.chdir(args.evalpath + '/' + dataset)

    # save tikz and pdf
    import tikzplotlib

    tikzplotlib.save(dataset + '_comparison.tex')
    plt.savefig(dataset + '_comparison.pdf')
    plt.show()

    # get mean of matrix-normal for plotting experts
    regcoeff = []
    for idx, m in enumerate(ilr.models):
        if idx in ilr.used_labels:
            M, _, _, _ = m.posterior.params
            regcoeff.append(M)

    # plot three experts
    plt.figure()
    axis = np.linspace(0, 1, 500).reshape(-1, 1)
    mu = []
    for t in range(len(axis)):
        q = np.hstack((axis[t, :], 1.))
        _mu = (regcoeff[0] @ q).tolist()
        mu.append(_mu)
    mu = np.asarray(mu).reshape(-1, 1)
    plt.plot(axis, mu, linewidth=2, c='green')

    mu = []
    for t in range(len(axis)):
        q = np.hstack((axis[t, :], 1.))
        _mu = (regcoeff[1] @ q).tolist()
        mu.append(_mu)
    mu = np.asarray(mu).reshape(-1, 1)
    plt.plot(axis, mu, linewidth=2, c='orange')

    mu = []
    for t in range(len(axis)):
        q = np.hstack((axis[t, :], 1.))
        _mu = (regcoeff[2] @ q).tolist()
        mu.append(_mu)
    mu = np.asarray(mu).reshape(-1, 1)
    plt.plot(axis, mu, linewidth=2, c='purple')

    # plot data
    plt.scatter(input, target, facecolors='none', edgecolors='k', linewidth=0.5)

    plt.ylabel('y')
    plt.xlabel('x')

    # save tikz and pdf
    import tikzplotlib

    tikzplotlib.save(dataset + '_experts.tex')
    plt.savefig(dataset + '_experts.pdf')

    plt.show()
