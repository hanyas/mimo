import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse

import numpy as np
import numpy.random as npr

import mimo
from mimo.distributions import NormalWishart
from mimo.distributions import GaussianWithNormalWishart

from mimo.distributions import MatrixNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishart

from mimo.distributions import TruncatedStickBreaking
from mimo.distributions import CategoricalWithStickBreaking

from mimo.mixtures import BayesianMixtureOfLinearGaussians

from tqdm import tqdm

import pathos
from pathos.pools import ProcessPool as Pool
nb_cores = pathos.multiprocessing.cpu_count()

from itertools import islice
import random


def _job(kwargs):
    ilr = kwargs.pop('model')
    args = kwargs.pop('arguments')

    for i in range(args.super_iters):
        if args.stochastic:
            # Stochastic meanfield VI
            ilr.meanfield_stochastic_descent(maxiter=args.svi_iters,
                                             stepsize=args.svi_stepsize,
                                             batchsize=args.svi_batchsize)

    return ilr


def parallel_ilr_inference(nb_jobs=50, models=None, **kwargs):
    kwargs_list = []
    for n in range(nb_jobs):
        kwargs['model'] = models[n]
        kwargs_list.append(kwargs.copy())

    with Pool(processes=min(nb_jobs, nb_cores),
              initializer=tqdm.set_lock,
              initargs=(tqdm.get_lock(),)) as p:
        ilrs = p.map(_job, kwargs_list)

    return ilrs


def batches(batch_size, data_size):
    idx_all = random.sample(range(data_size), data_size)
    idx_iter = iter(idx_all)
    yield from iter(lambda: list(islice(idx_iter, batch_size)), [])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate ilr with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/robot'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=5, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=50, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=100, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=25, type=int)
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=False)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)
    parser.add_argument('--nb_splits', help='number of dataset splits', default=15, type=int)

    args = parser.parse_args()

    import json
    print(json.dumps(vars(args), indent=4))

    np.random.seed(args.seed)

    input = np.load(args.datapath + '/ourwam4dof/icra2021/eight/wam_eight_train.npz')['input']
    target = np.load(args.datapath + '/ourwam4dof/icra2021/eight/wam_eight_train.npz')['target']

    # scale data
    # from sklearn.decomposition import PCA
    # input_transform = PCA(n_components=12, whiten=True)
    # target_transform = PCA(n_components=4, whiten=True)

    from sklearn.preprocessing import StandardScaler
    input_transform = StandardScaler()
    target_transform = StandardScaler()

    input_transform.fit(input)
    target_transform.fit(target)

    # define ilr models
    input_dim = input.shape[-1]
    target_dim = target.shape[-1]

    nb_params = input_dim
    if args.affine:
        nb_params += 1

    basis_prior, models_prior = [], []
    psi_nw, kappa = 1e-2, 1e-2
    psi_mnw, K = 1e2, 1e-2

    ilrs = []
    for k in range(args.nb_seeds):
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

        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)),
                                deltas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = TruncatedStickBreaking(**gating_hypparams)

        ilrs.append(BayesianMixtureOfLinearGaussians(gating=CategoricalWithStickBreaking(gating_prior),
                                                     basis=[GaussianWithNormalWishart(basis_prior[i]) for i in range(args.nb_models)],
                                                     models=[LinearGaussianWithMatrixNormalWishart(models_prior[i], affine=args.affine)
                                                             for i in range(args.nb_models)]))

    import copy
    from sklearn.utils import shuffle
    from sklearn.metrics import mean_squared_error, r2_score

    # split data for n sequential updates and k models
    data_size = input.shape[0]
    batch_size = int(data_size / args.nb_splits)

    mse = np.zeros((args.nb_seeds, args.nb_splits))
    smse = np.zeros((args.nb_seeds, args.nb_splits))
    nb_models = np.zeros((args.nb_seeds, args.nb_splits), dtype=np.int64)

    # iterate over different models
    for k, ilr in enumerate(ilrs):
        # shuffle data between models
        input, target = shuffle(input, target)

        # iterate over sequential updates
        for n in range(args.nb_splits):
            # clear all previous data
            if n > 0:
                ilr.clear_data()

            train_input = input[n * batch_size: (n + 1) * batch_size, :]
            train_target = target[n * batch_size: (n + 1) * batch_size, :]

            ilr.add_data(x=train_input, y=train_target,
                         whiten=True, transform_type='Standard',
                         target_transform=target_transform,
                         input_transform=input_transform)

            # init models
            if n == 0:
                ilr.resample(maxiter=1, progprint=args.verbose)

            # Meanfield VI
            ilr.meanfield_coordinate_descent(tol=args.earlystop,
                                             maxiter=args.meanfield_iters,
                                             progprint=args.verbose)

            # set posterior to prior
            ilr.gating.prior = copy.deepcopy(ilr.gating.posterior)
            for i in range(ilr.likelihood.size):
                ilr.basis[i].prior = copy.deepcopy(ilr.basis[i].posterior)
                ilr.models[i].prior = copy.deepcopy(ilr.models[i].posterior)

            nb_models[k, n] = len(ilr.used_labels)

            test_input = input[: (n + 1) * batch_size, :]
            test_target = target[: (n + 1) * batch_size, :]

            mu, _, _ = ilr.meanfield_prediction(x=test_input, prediction=args.prediction)

            mse[k, n] = mean_squared_error(test_target, mu)
            smse[k, n] = 1. - r2_score(test_target, mu)

            print('Seed', k, 'Iteration', n,
                  'MSE:', mse[k, n], 'SMSE:', smse[k, n],
                  'Compnents:', nb_models[k, n])

    import matplotlib.pyplot as plt
    import tikzplotlib

    plt.figure()
    plt.plot(mse[:5, :].T, marker='s')
    plt.yscale('log')
    plt.show()
    tikzplotlib.save("eight_sequential_mse.tex")

    plt.figure()
    plt.plot(smse[:5, :].T, marker='D')
    plt.yscale('log')
    plt.show()
    tikzplotlib.save("eight_sequential_smse.tex")
