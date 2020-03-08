import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, mixture
from mimo.util.text import progprint_xrange
from mimo.util.general import near_pd

import argparse

import matplotlib.pyplot as plt

import joblib
from joblib import Parallel, delayed
nb_cores = joblib.parallel.cpu_count()


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
    np.random.seed(seed)

    nb_params = input_dim
    if args.affine:
        nb_params += 1

    components_prior = []
    if args.kmeans:
        from sklearn.cluster import KMeans
        km = KMeans(args.nb_models).fit(np.hstack((input, target)))

        for n in range(args.nb_models):
            # initialize Normal
            mu_input = km.cluster_centers_[n, :input_dim]
            _psi_niw = np.cov(input[km.labels_ == n], bias=False, rowvar=False)
            psi_niw = np.diag(near_pd(np.atleast_2d(_psi_niw)))
            kappa = 1e-2

            # initialize Matrix-Normal
            mu_output = np.zeros((target_dim, nb_params))
            mu_output[:, -1] = km.cluster_centers_[n, input_dim:]
            psi_mniw = 1e0
            V = 1e3 * np.eye(nb_params)

            components_hypparams = dict(mu=mu_input, kappa=kappa,
                                        psi_niw=np.eye(input_dim) * psi_niw,
                                        nu_niw=input_dim + 1,
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
        V = 1e3 * np.eye(nb_params)

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
        dpglm = mixture.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                                components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
    else:
        dpglm = mixture.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                                components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
    dpglm.add_data(data)

    for _ in range(args.super_iters):
        # Gibbs sampling
        if args.verbose:
            print("Gibbs Sampling")

        gibbs_iter = range(args.gibbs_iters) if not args.verbose\
            else progprint_xrange(args.gibbs_iters)

        for _ in gibbs_iter:
            dpglm.resample_model()

        if args.stochastic:
            # Stochastic meanfield VI
            if args.verbose:
                print('Stochastic Variational Inference')

            svi_iter = range(args.gibbs_iters) if not args.verbose\
                else progprint_xrange(args.svi_iters)

            batch_size = args.svi_batchsize
            prob = batch_size / float(len(data))
            for _ in svi_iter:
                minibatch = npr.permutation(len(data))[:batch_size]
                dpglm.meanfield_sgdstep(minibatch=data[minibatch, :],
                                        prob=prob, stepsize=args.svi_stepsize)
        if args.deterministic:
            # Meanfield VI
            if args.verbose:
                print("Variational Inference")
            dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                               maxiter=args.meanfield_iters,
                                               progprint=args.verbose)

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
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=100, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=100, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=100, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=1000, type=int)
    parser.add_argument('--svi_iters', help='stochastic VI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='svi step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='svi batch size', default=256, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=True)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--nb_train', help='size of train dataset', default=2500, type=int)
    parser.add_argument('--nb_test', help='size of test dataset', default=500, type=int)
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # all data
    nb_data = args.nb_train + args.nb_test

    # create Sine data
    data = np.zeros((nb_data, 2))
    step = 10. * np.pi / nb_data
    for i in range(data.shape[0]):
        x = i * step
        data[i, 0] = x + npr.normal(0, .1) * 0.
        data[i, 1] = 3 * (np.sin(x) + npr.normal(0, .1))

    # shuffle data
    from sklearn.utils import shuffle
    data = shuffle(data)

    # training data
    nb_train = args.nb_train
    train_data = data[:nb_train, :]
    train_input, train_target = train_data[:, :1], train_data[:, 1:]

    # test data
    nb_test = args.nb_test
    test_data = data[-nb_test:, :]
    test_input, test_target = test_data[:, :1], test_data[:, 1:]

    # scale training data
    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=1, whiten=True)
    target_scaler = PCA(n_components=1, whiten=True)

    input_scaler.fit(data[:, :1])
    target_scaler.fit(data[:, 1:])

    train_data = {'input': input_scaler.transform(train_input),
                  'target': target_scaler.transform(train_target)}

    dpglm = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                     train_data=train_data,
                                     arguments=args)[0]

    # predict on training
    from mimo.util.prediction import meanfield_prediction

    train_mu, train_var, train_std, train_nlpd = [], [], [], []
    for t in range(len(train_input)):
        _mean, _var, _std, _nlpd = meanfield_prediction(dpglm, train_input[t, :],
                                                        train_target[t, :],
                                                        prediction=args.prediction,
                                                        input_scaler=input_scaler,
                                                        target_scaler=target_scaler)

        train_mu.append(_mean)
        train_var.append(_var)
        train_std.append(_std)
        train_nlpd.append(_nlpd)

    train_mu = np.hstack(train_mu)
    train_var = np.hstack(train_var)
    train_std = np.hstack(train_std)
    train_nlpd = np.hstack(train_nlpd)

    # metrics
    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
    mse = mean_squared_error(train_target, train_mu)
    evar = explained_variance_score(train_target, train_mu, multioutput='variance_weighted')
    smse = 1. - r2_score(train_target, train_mu, multioutput='variance_weighted')

    print('TRAIN - EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'NLPD:',
          train_nlpd.mean(), 'Compnents:', len(dpglm.used_labels))

    # plot prediction
    fig, axes = plt.subplots(2, 1)

    axes[0].scatter(train_input, train_mu + 2 * train_std, s=0.75, color='g')
    axes[0].scatter(train_input, train_mu - 2 * train_std, s=0.75, color='g')
    axes[0].scatter(train_input, train_mu, s=0.75, color='r')
    axes[0].scatter(train_input, train_target, s=0.75, color='k')

    axes[0].set_ylabel('y_train')

    # predict on testing
    test_mu, test_var, test_std, test_nlpd = [], [], [], []
    for t in range(len(test_input)):
        _mean, _var, _std, _nlpd = meanfield_prediction(dpglm, test_input[t, :],
                                                        test_target[t, :],
                                                        prediction=args.prediction,
                                                        input_scaler=input_scaler,
                                                        target_scaler=target_scaler)

        test_mu.append(_mean)
        test_var.append(_var)
        test_std.append(_std)
        test_nlpd.append(_nlpd)

    test_mu = np.hstack(test_mu)
    test_var = np.hstack(test_var)
    test_std = np.hstack(test_std)
    test_nlpd = np.hstack(test_nlpd)

    # metrics
    mse = mean_squared_error(test_target, test_mu)
    evar = explained_variance_score(test_target, test_mu, multioutput='variance_weighted')
    smse = 1. - r2_score(test_target, test_mu, multioutput='variance_weighted')

    print('TEST - EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'NLPD:',
          test_nlpd.mean(), 'Compnents:', len(dpglm.used_labels))

    axes[1].scatter(test_input, test_mu + 2 * test_std, s=0.75, color='g')
    axes[1].scatter(test_input, test_mu - 2 * test_std, s=0.75, color='g')
    axes[1].scatter(test_input, test_mu, s=0.75, color='r')
    axes[1].scatter(test_input, test_target, s=0.75, color='k')

    axes[1].set_ylabel('y_test')

    plt.show()
