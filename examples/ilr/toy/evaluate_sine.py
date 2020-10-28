import os
import argparse

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import mimo
from mimo.distributions import NormalGamma
from mimo.distributions import MatrixNormalWishart
from mimo.distributions import GaussianWithNormalGamma
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
    alpha_ng = 1.
    beta_ng = 1. / (1 * 1e2)
    kappas = 1e-2

    # initialize Matrix-Normal
    psi_mnw = 1e1
    K = 1e-2

    for n in range(args.nb_models):
        basis_hypparams = dict(mu=np.zeros((input_dim,)),
                               alphas=np.ones(input_dim) * alpha_ng,
                               betas=np.ones(input_dim) * beta_ng,
                               kappas=np.ones(input_dim) * kappas)

        aux = NormalGamma(**basis_hypparams)
        basis_prior.append(aux)

        models_hypparams = dict(M=np.zeros((target_dim, nb_params)),
                                K=np.eye(nb_params) * K, nu=target_dim + 1,
                                psi=np.eye(target_dim) * psi_mnw)

        aux = MatrixNormalWishart(**models_hypparams)
        models_prior.append(aux)

    # define gating
    if args.prior == 'stick-breaking':
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)),
                                deltas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = TruncatedStickBreaking(**gating_hypparams)

        ilr = BayesianMixtureOfLinearGaussians(gating=CategoricalWithStickBreaking(gating_prior),
                                               basis=[GaussianWithNormalGamma(basis_prior[i])
                                                      for i in range(args.nb_models)],
                                               models=[LinearGaussianWithMatrixNormalWishart(models_prior[i], affine=args.affine)
                                                       for i in range(args.nb_models)])

    else:
        gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = Dirichlet(**gating_hypparams)

        ilr = BayesianMixtureOfLinearGaussians(gating=CategoricalWithDirichlet(gating_prior),
                                               basis=[GaussianWithNormalGamma(basis_prior[i])
                                                      for i in range(args.nb_models)],
                                               models=[LinearGaussianWithMatrixNormalWishart(models_prior[i], affine=args.affine)
                                                       for i in range(args.nb_models)])
    ilr.add_data(target, input, whiten=True,
                 transform_type='Standard')

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
    parser.add_argument('--alpha', help='concentration parameter', default=50, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=50, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=3, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=10, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=25, type=int)
    parser.add_argument('--svi_iters', help='SVI iterations', default=100, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=1e-1, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=128, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--nb_train', help='size of train dataset', default=2000, type=int)
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # create Sine data
    nb_train = args.nb_train

    true_target = np.zeros((nb_train,))
    true_input = np.zeros((nb_train,))

    data = np.zeros((nb_train, 2))
    step = 10. * np.pi / nb_train
    for i in range(data.shape[0]):
        true_input[i] = i * step
        data[i, 0] = true_input[i] + 0.1 * npr.randn()
        true_target[i] = 3 * np.sin(true_input[i])
        data[i, 1] = true_target[i] + 0.3 * npr.randn()

    from itertools import chain

    r = list(chain(range(0, 500), range(1000, 1500)))
    train_data = data[r, :]

    # training data
    nb_train = args.nb_train
    input, target = data[:, :1], data[:, 1:]
    train_input, train_target = train_data[:, :1], train_data[:, 1:]

    ilr = parallel_ilr_inference(nb_jobs=args.nb_seeds,
                                 train_input=train_input,
                                 train_target=train_target,
                                 arguments=args)[0]

    # predict on training
    mu, var, std, nlpd = \
        ilr.meanfield_prediction(input, target, prediction=args.prediction)

    # metrics
    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

    mse = mean_squared_error(target, mu)
    evar = explained_variance_score(target, mu, multioutput='variance_weighted')
    smse = 1. - r2_score(target, mu, multioutput='variance_weighted')

    print('TRAIN - EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'NLPD:',
          nlpd.mean(), 'Compnents:', len(ilr.used_labels))

    fig, axes = plt.subplots(2, 1)

    # plot prediction
    sorter = np.argsort(input, axis=0).flatten()
    sorted_input, sorted_target = input[sorter, 0], target[sorter, 0]
    sorted_mu, sorted_std = mu[sorter, 0], std[sorter, 0]

    axes[0].plot(true_input, true_target, '--k')
    axes[0].scatter(train_input, train_target, marker='+', s=1.25, color='k')
    axes[0].plot(sorted_input, sorted_mu, color='crimson')
    for c in [1., 2., 3.]:
        axes[0].fill_between(sorted_input,
                             sorted_mu - c * sorted_std,
                             sorted_mu + c * sorted_std,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1))

    axes[0].set_ylabel('y')
    axes[0].set_ylim(-15., 15.)

    # plot gaussian activations
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p(x)')

    activations = ilr.meanfield_predictive_activation(sorted_input)
    axes[1].plot(sorted_input, activations)

    plt.show()
