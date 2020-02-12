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
    np.random.seed(seed=seed)

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
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation_uai2020'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=25, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=50, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=100, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--deterministic', help='use deterministic VI', dest='stochastic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=750, type=int)
    parser.add_argument('--svi_iters', help='stochastic VI iterations', default=2500, type=int)
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
    _input = np.linspace(-10., 10., nb_samples).reshape(nb_samples, 1)
    noise = lambda x: 0.05 + 0.2 * (1. + np.sin(2. * x)) / (1. + np.exp(-0.2 * x))

    mu_predict_list, std_predict_list, evar_list, mse_list, smse_list = [], [], [], [], []
    for i in range(10):

        np.random.seed()

        # subsample dataset
        rows = np.random.choice(_input.shape[0], 4000)
        _input = _input[rows, :]
        sorting = np.argsort(_input, axis=0)  # sort based on input values
        input = np.take_along_axis(_input, sorting, axis=0)

        target = np.sinc(input) + noise(input) * np.random.randn(len(input), 1)
        mean = np.sinc(input)

        # plt.figure()
        # plt.plot(input, mean, '--b')
        # plt.plot(input, mean + 2 * noise(input), '--g')
        # plt.plot(input, mean - 2 * noise(input), '--g')
        # plt.scatter(input, target, s=0.75, c='k')

        # # Original Data
        # train_data = {'input': input, 'target': target}
        #
        # dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
        #                                   train_data=train_data,
        #                                   arguments=args)
        #
        # from mimo.util.prediction import meanfield_prediction
        #
        # mu_predic, std_predict = [], []
        # for t in range(len(input)):
        #     _mu, _, _std = meanfield_prediction(dpglms[0], input[t, :])
        #     mu_predic.append(_mu)
        #     std_predict.append(_std)
        #
        # mu_predic = np.vstack(mu_predic)
        # std_predict = np.vstack(std_predict)
        #
        # plt.plot(train_data['input'], mu_predic, '-c')
        # plt.plot(train_data['input'], mu_predic + 2 * std_predict, '-r')
        # plt.plot(train_data['input'], mu_predic - 2 * std_predict, '-r')
        #
        # plt.figure()
        # plt.plot(std_predict)
        # plt.plot(noise(input))

        # Scaled Data
        from sklearn.decomposition import PCA
        input_scaler = PCA(n_components=1, whiten=True)
        target_scaler = PCA(n_components=1, whiten=True)

        input_scaler.fit(input)
        target_scaler.fit(target)

        scaled_train_data = {'input': input_scaler.transform(input),
                             'target': target_scaler.transform(target)}

        dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                          train_data=scaled_train_data,
                                          arguments=args)

        # predict
        from mimo.util.prediction import parallel_meanfield_prediction
        mu_predict, var_predict, std_predict = parallel_meanfield_prediction(dpglms[0], input,
                                                                             prediction=args.prediction,
                                                                             input_scaler=input_scaler,
                                                                             target_scaler=target_scaler)

        from sklearn.metrics import explained_variance_score, mean_squared_error
        evar = explained_variance_score(mu_predict, target)
        mse = mean_squared_error(mu_predict, target)
        smse = mean_squared_error(mu_predict, target) / np.var(target, axis=0)

        print('EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'Compnents:', len(dpglms[0].used_labels))

        # plt.plot(input, mu_predict, '-m')
        # plt.plot(input, mu_predict + 2 * std_predict, '-r')
        # plt.plot(input, mu_predict - 2 * std_predict, '-r')
        #
        # plt.figure()
        # plt.plot(std_predict)
        # plt.plot(noise(input))
        #
        # plt.show()

        mu_predict_list.append(mu_predict)
        std_predict_list.append(std_predict)
        evar_list.append(evar_list)
        mse_list.append(mse_list)
        smse_list.append(smse_list)

    # calcule means and confidence intervals
    mu_predict_avg = sum(mu_predict_list) / len(mu_predict_list)
    mu_predict_std = (sum([((x - mu_predict_avg) ** 2) for x in mu_predict_list]) / len(mu_predict_list)) ** 0.5
    std_predict_avg = sum(std_predict_list) / len(std_predict_list)
    std_predict_std = (sum([((x - std_predict_avg) ** 2) for x in std_predict_list]) / len(std_predict_list)) ** 0.5

    # plot prediction, gaussian activations and noise levels in one plot
    from matplotlib import gridspec
    import scipy.stats as stats

    # create figure
    w, h = plt.figaspect(0.67) #figure is wider than tall
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
    x_mu, x_sigma = [], []
    for idx, c in enumerate(dpglms[0].components):
        if idx in dpglms[0].used_labels:
            mu, kappa, psi_niw, _, _, _, _, _ = c.posterior.params

            mu = input_scaler.inverse_transform(np.atleast_2d(mu))
            trans = np.sqrt(input_scaler.explained_variance_[:, None])
            psi_niw = trans.T @ psi_niw @ trans

            sigma = np.sqrt(1 / kappa * psi_niw)
            x_mu.append(mu[0])
            x_sigma.append(sigma[0])

    for i in range(len(dpglms[0].used_labels)):
        x = np.linspace(-10, 10, 200)
        ax1.plot(x, stats.norm.pdf(x, x_mu[i], x_sigma[i]), 'k-')

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'sinc'

    # save figs
    import tikzplotlib
    path = os.path.join(str(dataset))
    tikzplotlib.save(path + '_example.tex')
    plt.savefig(path + '_example.pdf')
    # plt.show()


    # plot mean and standard deviation of mean estimation
    w, h = plt.figaspect(0.67) #figure is wider than tall
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
    ax3.plot(std_predict_avg, '-r')
    ax3.plot(std_predict_avg + 2 * std_predict_std, '-b')
    ax3.plot(std_predict_avg - 2 * std_predict_std, '-b')
    ax3.plot(noise(input), 'k--')

    # plt.gridSpec.tightlayout()

    # save time stamp for file names
    import datetime
    time = str(datetime.datetime.now().strftime('_%m-%d_%H-%M-%S'))

    # save figs
    path = os.path.join(str(dataset))
    tikzplotlib.save(path + '_mean.tex')
    plt.savefig(path + '_mean.pdf')
    # plt.show()
