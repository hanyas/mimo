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

from mimo.distributions import Dirichlet
from mimo.distributions import CategoricalWithDirichlet

from mimo.mixtures import BayesianMixtureOfLinearGaussians

from tqdm import tqdm

import pathos
from pathos.pools import ProcessPool as Pool
nb_cores = pathos.multiprocessing.cpu_count()


def _job(kwargs):
    args = kwargs.pop('arguments')
    seed = kwargs.pop('seed')

    input = kwargs.pop('input')
    target = kwargs.pop('target')

    input_transform = kwargs.pop('input_transform')
    target_transform = kwargs.pop('target_transform')

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
    psi_nw = 5e-3
    kappa = 1e-2

    # initialize Matrix-Normal
    psi_mnw = 5e-3
    K = 1e-2

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
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)),
                                deltas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = TruncatedStickBreaking(**gating_hypparams)

        ilr = BayesianMixtureOfLinearGaussians(gating=CategoricalWithStickBreaking(gating_prior),
                                               basis=[GaussianWithNormalWishart(basis_prior[i]) for i in range(args.nb_models)],
                                               models=[LinearGaussianWithMatrixNormalWishart(models_prior[i], affine=args.affine)
                                                       for i in range(args.nb_models)])

    else:
        gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = Dirichlet(**gating_hypparams)

        ilr = BayesianMixtureOfLinearGaussians(gating=CategoricalWithDirichlet(gating_prior),
                                               basis=[GaussianWithNormalWishart(basis_prior[i]) for i in range(args.nb_models)],
                                               models=[LinearGaussianWithMatrixNormalWishart(models_prior[i], affine=args.affine)
                                                       for i in range(args.nb_models)])

    ilr.add_data(target, input, whiten=True,
                 transform_type='Standard',
                 target_transform=target_transform,
                 input_transform=input_transform)

    # Gibbs sampling
    ilr.resample(maxiter=args.gibbs_iters,
                 progprint=args.verbose)

    for k in range(args.super_iters):
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

    return ilr


def parallel_ilr_inference(nb_jobs=50, **kwargs):
    kwargs_list = []
    for n in range(nb_jobs):
        kwargs['seed'] = npr.randint(1337, 6174)
        kwargs_list.append(kwargs.copy())

    with Pool(processes=min(nb_jobs, nb_cores),
              initializer=tqdm.set_lock,
              initargs=(tqdm.get_lock(),)) as p:
        res = p.map(_job, kwargs_list)

    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate ilr with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/robot'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=500, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=1000, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=5, type=int)
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=25, type=int)
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--svi_iters', help='SVI iterations', default=10, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=1e-3, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=1024, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    import json
    print(json.dumps(vars(args), indent=4))

    np.random.seed(args.seed)

    train_input = np.load(args.datapath + '/ourwam4dof/icra2021/invdyn/wam_invdyn_train.npz')['input']
    train_target = np.load(args.datapath + '/ourwam4dof/icra2021/invdyn/wam_invdyn_train.npz')['target']

    test_input = np.load(args.datapath + '/ourwam4dof/icra2021/invdyn/wam_invdyn_test.npz')['input']
    test_target = np.load(args.datapath + '/ourwam4dof/icra2021/invdyn/wam_invdyn_test.npz')['target']

    # train_input = np.load(args.datapath + '/ourwam4dof/icra2021/eight/wam_eight_train.npz')['input']
    # train_target = np.load(args.datapath + '/ourwam4dof/icra2021/eight/wam_eight_train.npz')['target']
    #
    # test_input = np.load(args.datapath + '/ourwam4dof/icra2021/eight/wam_eight_test.npz')['input']
    # test_target = np.load(args.datapath + '/ourwam4dof/icra2021/eight/wam_eight_test.npz')['target']

    input_data = np.vstack((train_input, test_input))
    target_data = np.vstack((train_target, test_target))

    # # scale data
    # from sklearn.decomposition import PCA
    # input_transform = PCA(n_components=12, whiten=True)
    # target_transform = PCA(n_components=4, whiten=True)

    from sklearn.preprocessing import StandardScaler
    input_transform = StandardScaler()
    target_transform = StandardScaler()

    input_transform.fit(input_data)
    target_transform.fit(target_data)

    ilrs = parallel_ilr_inference(nb_jobs=args.nb_seeds,
                                  input=train_input,
                                  target=train_target,
                                  input_transform=input_transform,
                                  target_transform=target_transform,
                                  arguments=args)

    from sklearn.metrics import mean_squared_error, r2_score

    test_mse, test_smse, test_nlpd, nb_models = [], [], [], []
    for ilr in ilrs:
        _nb_models = len(ilr.used_labels)

        # _train_mu, _, _, _train_nlpd = \
        #     ilr.meanfield_prediction(x=train_input,often, you
        #                              y=train_target,
        #                              prediction=args.prediction)
        #
        # _train_mse = mean_squared_error(train_target, _train_mu)
        # _train_smse = 1. - r2_score(train_target, _train_mu)
        #
        # print('TRAIN - MSE:', _train_mse, 'SMSE:', _train_smse,
        #       'NLPD:', _train_nlpd.mean(), 'Compnents:', _nb_models)

        _test_mu, _, _, _test_nlpd =\
            ilr.meanfield_prediction(x=test_input,
                                     y=test_target,
                                     prediction=args.prediction)

        _test_mse = mean_squared_error(test_target, _test_mu)
        _test_smse = 1. - r2_score(test_target, _test_mu)

        print('TEST - MSE:', _test_mse, 'SMSE:', _test_smse,
              'NLPD:', _test_nlpd.mean(), 'Compnents:', _nb_models)

        test_mse.append(_test_mse)
        test_smse.append(_test_smse)
        test_nlpd.append(_test_nlpd.mean())
        nb_models.append(_nb_models)

    mean_mse = np.mean(test_mse)
    std_mse = np.std(test_mse)

    mean_smse = np.mean(test_smse)
    std_smse = np.std(test_smse)

    mean_nlpd = np.mean(test_nlpd)
    std_nlpd = np.std(test_nlpd)

    mean_nb_models = np.mean(nb_models)
    std_nb_models = np.std(nb_models)

    arr = np.array([mean_mse, std_mse,
                    mean_smse, std_smse,
                    mean_nlpd, std_nlpd,
                    mean_nb_models, std_nb_models])

    # import pandas as pd
    # dt = pd.DataFrame(data=arr, index=['mse_avg', 'mse_std',
    #                                    'smse_avg', 'smse_std',
    #                                    'nlpd_avg', 'nlpd_std',
    #                                    'models_avg', 'models_std'])
    #
    # dt.to_csv('wam_' + str(args.prior) +
    #           '_alpha_' + str(args.alpha) + '.csv',
    #           mode='a', index=True)

    # # # save model
    # ilrs[0].clear_data()
    #
    # import pickle
    # pickle.dump(ilrs[0], open("wam_invdyn_0.p", "wb"))
