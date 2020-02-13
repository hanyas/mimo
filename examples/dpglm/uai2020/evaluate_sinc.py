import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, models
from mimo.util.text import progprint_xrange
from mimo.util.general import near_pd

import argparse

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing
nb_cores = multiprocessing.cpu_count()


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
            _psi_input = np.cov(input[km.labels_ == n], bias=False, rowvar=False)
            psi_input = near_pd(np.atleast_2d(_psi_input))
            kappa = 1e-2

            # initialize Matrix-Normal
            mu_output = np.zeros((target_dim, nb_params))
            mu_output[:, -1] = km.cluster_centers_[n, input_dim:]
            psi_mniw = 1e0
            V = 1e3 * np.eye(nb_params)

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
        if args.verbose:
            print("Gibbs Sampling")
        for _ in gibbs_iter:
            dpglm.resample_model()

        if not args.stochastic:
            # Meanfield VI
            if args.verbose:
                print("Variational Inference")
            dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                               maxiter=args.meanfield_iters,
                                               progprint=args.verbose)
        else:
            svi_iter = range(args.gibbs_iters) if not args.verbose\
                else progprint_xrange(args.svi_iters)

            # Stochastic meanfield VI
            if args.verbose:
                print('Stochastic Variational Inference')
            batch_size = args.svi_batchsize
            prob = batch_size / float(len(data))
            for _ in svi_iter:
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
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=50, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=50, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=100, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--deterministic', help='use deterministic VI', dest='stochastic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=750, type=int)
    parser.add_argument('--svi_iters', help='stochastic VI iterations', default=1000, type=int)
    parser.add_argument('--svi_stepsize', help='svi step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='svi batch size', default=256, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=True)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')

    args = parser.parse_args()

    np.random.seed(1337)

    # sample dataset
    nb_samples = 5000
    input = np.linspace(-10., 10., nb_samples).reshape(nb_samples, 1)
    noise = lambda x: 0.05 + 0.2 * (1. + np.sin(2. * x)) / (1. + np.exp(-0.2 * x))
    target = np.sinc(input) + noise(input) * np.random.randn(len(input), 1)
    mean = np.sinc(input)

    # Data scaling
    from sklearn.decomposition import PCA

    input_scaler = PCA(n_components=1, whiten=True)
    target_scaler = PCA(n_components=1, whiten=True)

    input_scaler.fit(input)
    target_scaler.fit(target)

    scaled_data = {'input': input_scaler.transform(input),
                   'target': target_scaler.transform(target)}

    dpglm = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                     train_data=scaled_data, arguments=args)[0]

    # predict
    from mimo.util.prediction import parallel_meanfield_prediction

    mu_predict, var_predict, std_predict = parallel_meanfield_prediction(dpglm, input,
                                                                         prediction=args.prediction,
                                                                         input_scaler=input_scaler,
                                                                         target_scaler=target_scaler)

    from sklearn.metrics import explained_variance_score, mean_squared_error

    evar = explained_variance_score(mu_predict, target)
    mse = mean_squared_error(mu_predict, target)
    smse = mean_squared_error(mu_predict, target) / np.var(target, axis=0)

    # plot prediction, gaussian activations and noise levels in one plot
    from matplotlib import gridspec
    import scipy.stats as stats

    # create figure
    w, h = plt.figaspect(0.67)  # figure is wider than tall
    fig = plt.figure(figsize=(w, h))
    gs1 = gridspec.GridSpec(2, 1, height_ratios=[6, 2])
    ax0 = fig.add_subplot(gs1[0])
    ax1 = fig.add_subplot(gs1[1])

    # plot data and prediction
    ax0.plot(input, mean, 'k--')
    ax0.plot(input, mean + 2 * noise(input), 'g--')
    ax0.plot(input, mean - 2 * noise(input), 'g--')
    ax0.scatter(input, target, s=0.75, facecolors='none', edgecolors='grey')

    ax0.plot(input, mu_predict, '-r')
    ax0.plot(input, mu_predict + 2 * std_predict, '-b')
    ax0.plot(input, mu_predict - 2 * std_predict, '-b')

    # plot gaussian activations
    mu, sigma = [], []
    for idx, c in enumerate(dpglm.components):
        if idx in dpglm.used_labels:
            _mu, _sigma, _, _ = c.posterior.mode()

            _mu = input_scaler.inverse_transform(np.atleast_2d(_mu))
            trans = (np.sqrt(input_scaler.explained_variance_[:, None]) * input_scaler.components_).T
            _sigma = trans.T @ np.diag(_sigma) @ trans

            mu.append(_mu)
            sigma.append(_sigma)

    activations = []
    for i in range(len(dpglm.used_labels)):
        activations.append(stats.norm.pdf(input, mu[i], np.sqrt(sigma[i])))

    activations = np.asarray(activations).squeeze()
    # activations = activations / np.sum(activations, axis=1, keepdims=True)
    activations = activations / np.sum(activations, axis=0, keepdims=True)

    for i in range(len(dpglm.used_labels)):
        ax1.plot(input, activations[i])

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'sinc'

    # save figs
    import tikzplotlib
    path = os.path.join(str(dataset))
    tikzplotlib.save(path + '_example.tex')
    plt.savefig(path + '_example.pdf')
    plt.show()

    mu_predict_list, std_predict_list,  = [], []
    evar_list, mse_list, smse_list = [], [], []
    for i in range(25):
        np.random.seed(i)

        # subsample dataset
        rows = np.random.choice(input.shape[0], 4000)
        _input = input[rows, :]
        _target = np.sinc(_input) + noise(_input) * np.random.randn(len(_input), 1)

        _scaled_data = {'input': input_scaler.transform(_input),
                        'target': target_scaler.transform(_target)}

        dpglm = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                         train_data=_scaled_data,
                                         arguments=args)[0]

        # predict
        from mimo.util.prediction import parallel_meanfield_prediction
        _mu_predict, _var_predict, _std_predict = parallel_meanfield_prediction(dpglm, input,
                                                                                prediction=args.prediction,
                                                                                input_scaler=input_scaler,
                                                                                target_scaler=target_scaler)

        _evar = explained_variance_score(_mu_predict, target)
        _mse = mean_squared_error(_mu_predict, target)
        _smse = mean_squared_error(_mu_predict, target) / np.var(target, axis=0)

        print('EVAR:', _evar, 'MSE:', _mse, 'SMSE:', _smse, 'Compnents:', len(dpglm.used_labels))

        mu_predict_list.append(_mu_predict)
        std_predict_list.append(_std_predict)
        evar_list.append(_evar)
        mse_list.append(_mse)
        smse_list.append(_smse)

    mu_predict_list = np.asarray(mu_predict_list).squeeze()
    std_predict_list = np.asarray(std_predict_list).squeeze()

    # calcule means and confidence intervals
    mu_predict_avg, mu_predict_std = np.mean(mu_predict_list, axis=0), np.std(mu_predict_list, axis=0)
    std_predict_avg, std_predict_std = np.mean(std_predict_list, axis=0), np.std(std_predict_list, axis=0)

    # plot mean and standard deviation of mean estimation
    w, h = plt.figaspect(0.67)  # figure is wider than tall
    fig = plt.figure(figsize=(w, h))
    gs2 = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax2 = fig.add_subplot(gs2[0])
    ax3 = fig.add_subplot(gs2[1])

    # ax2.scatter(input, target, s=0.75, color='k', alpha=0.5)
    ax2.scatter(input, target, s=0.75, facecolors='none', edgecolors='grey')
    ax2.plot(input, mu_predict_avg, '-r')
    ax2.plot(input, mu_predict_avg + 2 * mu_predict_std, '-b')
    ax2.plot(input, mu_predict_avg - 2 * mu_predict_std, '-b')

    # plot mean and standard deviation of data generation / estimated noise level
    ax3.plot(input, std_predict_avg, '-r')
    ax3.plot(input, std_predict_avg + 2 * std_predict_std, '-b')
    ax3.plot(input, std_predict_avg - 2 * std_predict_std, '-b')
    ax3.plot(input, noise(input), 'k--')

    plt.tight_layout()

    # save time stamp for file names
    import datetime
    time = str(datetime.datetime.now().strftime('_%m-%d_%H-%M-%S'))

    # save figs
    path = os.path.join(str(dataset))
    tikzplotlib.save(path + '_mean.tex')
    plt.savefig(path + '_mean.pdf')
    # plt.show()
