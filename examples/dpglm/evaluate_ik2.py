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

from mimo.distributions import StickBreaking
from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet
from mimo.distributions import CategoricalWithStickBreaking

from mimo.mixtures import BayesianMixtureOfLinearGaussians

import matplotlib.pyplot as plt
from tqdm import tqdm

import pathos
from pathos.pools import _ProcessPool as Pool
nb_cores = pathos.multiprocessing.cpu_count()


def generate_2d_kinematics(grid=(10, 10), l1=1.0, l2=1.):

    # joint angles
    q1 = np.linspace(-np.pi, np.pi, grid[0])
    q2 = np.linspace(-np.pi, np.pi, grid[1])

    q1, q2 = np.meshgrid(q1, q2)

    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    return np.vstack((q1.flatten(), q2.flatten(),
                      x.flatten(), y.flatten())).T


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
    psi_nw = 1e0
    kappa = 1e-2

    # initialize Matrix-Normal
    psi_mnw = 1e0
    K = 1e-3

    for n in range(args.nb_models):
        basis_hypparams = dict(mu=np.zeros((input_dim, )),
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
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)), deltas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = StickBreaking(**gating_hypparams)

        dpglm = BayesianMixtureOfLinearGaussians(gating=CategoricalWithStickBreaking(gating_prior),
                                                 basis=[GaussianWithNormalWishart(basis_prior[i])
                                                        for i in range(args.nb_models)],
                                                 models=[LinearGaussianWithMatrixNormalWishart(models_prior[i], affine=args.affine)
                                                         for i in range(args.nb_models)])

    else:
        gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = Dirichlet(**gating_hypparams)

        dpglm = BayesianMixtureOfLinearGaussians(gating=CategoricalWithDirichlet(gating_prior),
                                                 basis=[GaussianWithNormalWishart(basis_prior[i])
                                                        for i in range(args.nb_models)],
                                                 models=[LinearGaussianWithMatrixNormalWishart(models_prior[i], affine=args.affine)
                                                         for i in range(args.nb_models)])
    dpglm.add_data(target, input, whiten=True)

    # Gibbs sampling
    dpglm.resample(maxiter=args.gibbs_iters,
                   progprint=args.verbose)

    for _ in range(args.super_iters):
        if args.stochastic:
            # Stochastic meanfield VI
            dpglm.meanfield_stochastic_descent(maxiter=args.svi_iters,
                                               stepsize=args.svi_stepsize,
                                               batchsize=args.svi_batchsize)
        if args.deterministic:
            # Meanfield VI
            dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                               maxiter=args.meanfield_iters,
                                               progprint=args.verbose)

        dpglm.gating.prior = dpglm.gating.posterior
        for i in range(dpglm.size):
            dpglm.basis[i].prior = dpglm.basis[i].posterior
            dpglm.models[i].prior = dpglm.models[i].posterior

    return dpglm


def parallel_dpglm_inference(nb_jobs=50, **kwargs):
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

    parser = argparse.ArgumentParser(description='Evaluate DPGLM with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=250, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=100, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=25, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=1000, type=int)
    parser.add_argument('--svi_iters', help='SVI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=256, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='mode')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    data = generate_2d_kinematics(grid=(25, 25))
    input, target = data[:, 2:], data[:, :2]

    plt.figure()
    plt.scatter(input[:, 0], input[:, 1], s=1)

    dpglm = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                     train_input=input,
                                     train_target=target,
                                     arguments=args)[0]

    # predict
    mu, var, std = dpglm.meanfield_prediction(input, prediction=args.prediction)

    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

    mse = mean_squared_error(target, mu)
    evar = explained_variance_score(target, mu, multioutput='variance_weighted')
    smse = 1. - r2_score(target, mu, multioutput='variance_weighted')

    print('EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'Compnents:', len(dpglm.used_labels))

    plt.figure()
    plt.scatter(target[:, 0], target[:, 1], s=1)
    plt.scatter(mu[:, 0], mu[:, 1], s=1, c='r')

    plt.show()
