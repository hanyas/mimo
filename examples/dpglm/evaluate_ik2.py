import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, models
from mimo.util.text import progprint_xrange
from mimo.util.general import near_pd

import os
import argparse

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing
nb_cores = multiprocessing.cpu_count()


def generate_2d_kinematics(grid=(10, 10), l1=1., l2=1., seed=1337):

    # set random seed
    np.random.seed(seed=seed)

    # joint angles
    q1 = np.linspace(0., np.pi, grid[0])
    q2 = np.linspace(0., np.pi, grid[1])

    q1, q2 = np.meshgrid(q1, q2)

    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    return np.vstack((q1.flatten(), q2.flatten(),
                      x.flatten(), y.flatten())).T


def create_job(kwargs):
    train_data = kwargs.pop('train_data')
    args = kwargs.pop('arguments')
    seed = kwargs.pop('seed')

    input = train_data['input']
    target = train_data['target']
    data = np.hstack((input, target))

    input_dim = input.shape[-1]
    target_dim = target.shape[-1]

    # set random seed
    np.random.seed(seed=seed)

    nb_params = input_dim
    if args.affine:
        nb_params += 1

    components_prior = []
    if args.init_kmeans:
        from sklearn.cluster import KMeans
        km = KMeans(args.nb_models).fit(np.hstack((input, target)))

        for n in range(args.nb_models):
            # initialize Normal
            mu_input = km.cluster_centers_[n, :input_dim]
            _psi_input = np.cov(input[km.labels_ == n], bias=False, rowvar=False)
            psi_input = near_pd(np.atleast_2d(_psi_input))
            kappa = 1e-2

            # initialize Matrix-Normal
            mu_output = np.zeros((target_dim, nb_params))
            mu_output[:, -1] = km.cluster_centers_[n, input_dim:]
            psi_mniw = 1e0
            V = 1e12 * np.eye(nb_params)

            components_hypparams = dict(mu=mu_input, kappa=kappa,
                                        psi_niw=psi_input, nu_niw=input_dim + 1,
                                        M=mu_output, affine=args.affine,
                                        V=V, nu_mniw=target_dim + 1,
                                        psi_mniw=np.eye(target_dim) * psi_mniw)

            aux = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)
            components_prior.append(aux)
    else:
        # initialize Normal
        mu_low = np.min(input, axis=0)
        mu_high = np.max(input, axis=0)
        psi_niw = 1e0
        kappa = 1e-2

        # initialize Matrix-Normal
        psi_mniw = 1e0
        V = 1e12 * np.eye(nb_params)

        for n in range(args.nb_models):
            components_hypparams = dict(mu=npr.uniform(mu_low, mu_high, size=input_dim),
                                        kappa=kappa, psi_niw=np.eye(input_dim) * psi_niw,
                                        nu_niw=input_dim + 1,
                                        M=np.zeros((target_dim, nb_params)),
                                        affine=args.affine, V=V,
                                        nu_mniw=target_dim + 1,
                                        psi_mniw=np.eye(target_dim) * psi_mniw)

            aux = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)
            components_prior.append(aux)

    # define gating
    if args.prior == 'stick-breaking':
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)), deltas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = distributions.StickBreaking(**gating_hypparams)
    else:
        gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = distributions.Dirichlet(**gating_hypparams)

    # define model
    if args.prior == 'stick-breaking':
        dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                               components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
    else:
        dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                               components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
    dpglm.add_data(data)

    for _ in range(args.super_iters):
        gibbs_iter = range(args.gibbs_iters) if not args.verbose\
            else progprint_xrange(args.gibbs_iters)

        # Gibbs sampling
        print("Gibbs Sampling")
        for _ in gibbs_iter:
            dpglm.resample_model()

        if not args.stochastic:
            # Meanfield VI
            print("Variational Inference")
            dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                               maxiter=args.meanfield_iters,
                                               progprint=args.verbose)
        else:
            svi_iters = range(args.gibbs_iters) if not args.verbose\
                else progprint_xrange(args.svi_iters)

            # Stochastic meanfield VI
            print('Stochastic Variational Inference')
            batch_size = args.svi_batchsize
            prob = batch_size / float(len(data))
            for _ in svi_iters:
                minibatch = npr.permutation(len(data))[:batch_size]
                dpglm.meanfield_sgdstep(minibatch=data[minibatch, :],
                                        prob=prob, stepsize=args.svi_stepsize)

    return dpglm


def parallel_dpglm_inference(nb_jobs=50, **kwargs):
    kwargs_list = []
    for n in range(nb_jobs):
        kwargs['seed'] = n
        kwargs_list.append(kwargs.copy())

    return Parallel(n_jobs=min(nb_jobs, nb_cores),
                    verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate DPGLM with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=100, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=50, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default='True')
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=100, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default='True')
    parser.add_argument('--deterministic', help='use deterministic VI', dest='stochastic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=500, type=int)
    parser.add_argument('--svi_iters', help='stochastic VI iterations', default=2500, type=int)
    parser.add_argument('--svi_stepsize', help='svi step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='svi batch size', default=256, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default='True')
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default='True')
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')

    args = parser.parse_args()

    np.random.seed(1337)

    data = generate_2d_kinematics(grid=(50, 50))
    input, target = data[:, 2:], data[:, :2]

    plt.figure()
    plt.scatter(input[:, 0], input[:, 1], s=1)

    # # Original Data
    # train_data = {'input': input, 'target': target}
    #
    # dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
    #                                   train_data=train_data,
    #                                   arguments=args)
    #
    # from mimo.util.prediction import meanfield_prediction
    #
    # mu_predict = []
    # for t in range(len(input)):
    #     _mu, _, _ = meanfield_prediction(dpglms[0], input[t, :])
    #     mu_predict.append(_mu)
    #
    # mu_predict = np.vstack(mu_predict)
    #
    # plt.plot(train_data['input'], mu_predict, '-c')
    #
    # from sklearn.metrics import explained_variance_score, mean_squared_error
    # evar = explained_variance_score(mu_predict, target)
    # mse = mean_squared_error(mu_predict, target)
    #
    # smse = mean_squared_error(mu_predict, target) / np.var(target, axis=0)
    #
    # print('EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'Compnents:', len(dpglms[0].used_labels))
    #
    # plt.figure()
    # plt.scatter(target[:, 0], target[:, 1], s=1)
    # plt.scatter(mu_predict[:, 0], mu_predict[:, 1], s=1, c='r')

    # Scaled Data
    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=2, whiten=True)
    target_scaler = PCA(n_components=2, whiten=True)

    input_scaler.fit(input)
    target_scaler.fit(target)

    scaled_input = input_scaler.transform(input)
    scaled_target = target_scaler.transform(target)

    scaled_train_data = {'input': scaled_input,
                         'target': scaled_target}

    dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                      train_data=scaled_train_data,
                                      arguments=args)

    # predict
    from mimo.util.prediction import meanfield_prediction

    mu_predict = []
    for t in range(len(scaled_input)):
        _mean, _, _ = meanfield_prediction(dpglms[0], scaled_input[t, :], prediction='mode')
        mu_predict.append(target_scaler.inverse_transform(np.atleast_2d(_mean)))

    mu_predict = np.vstack(mu_predict)

    from sklearn.metrics import explained_variance_score, mean_squared_error
    evar = explained_variance_score(mu_predict, target)
    mse = mean_squared_error(mu_predict, target)

    smse = mean_squared_error(mu_predict, target) / np.var(target, axis=0)

    print('EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'Compnents:', len(dpglms[0].used_labels))

    plt.figure()
    plt.scatter(target[:, 0], target[:, 1], s=1)
    plt.scatter(mu_predict[:, 0], mu_predict[:, 1], s=1, c='r')