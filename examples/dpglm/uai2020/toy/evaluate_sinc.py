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

import pathos
from pathos.pools import _ProcessPool as Pool
nb_cores = pathos.multiprocessing.cpu_count()


def _job(kwargs):
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
            # _psi_niw = np.cov(input[km.labels_ == n], bias=False, rowvar=False)
            psi_niw = 1e-2  # np.diag(near_pd(np.atleast_2d(_psi_niw)))
            kappa = 1e-2

            # initialize Matrix-Normal
            mu_output = np.zeros((target_dim, nb_params))
            mu_output[:, -1] = km.cluster_centers_[n, input_dim:]
            psi_mniw = 1e0
            if args.affine:
                X = np.hstack((input, np.ones((len(input), 1))))
            else:
                X = input

            V = 50 * X.T @ X

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
        psi_niw = 1e-1
        kappa = 1e-2  # (1. / (input.T @ input)).item()

        # initialize Matrix-Normal
        psi_mniw = 1e-1
        if args.affine:
            X = np.hstack((input, np.ones((len(input), 1))))
        else:
            X = input

        V = X.T @ X

        for n in range(args.nb_models):
            components_hypparams = dict(mu=np.zeros((input_dim, )),
                                        kappa=kappa, psi_niw=np.eye(input_dim) * psi_niw,
                                        nu_niw=input_dim + 1,
                                        M=np.zeros((target_dim, nb_params)),
                                        affine=args.affine, V=V,
                                        nu_mniw=target_dim + 37,
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
                                components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i])
                                            for i in range(args.nb_models)])
    else:
        dpglm = mixture.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                                components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i])
                                            for i in range(args.nb_models)])
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
        _kwargs = {'seed': kwargs['arguments'].seed,
                   'train_data': kwargs['train_data'][n],
                   'arguments': kwargs['arguments']}
        kwargs_list.append(_kwargs)

    with Pool(processes=min(nb_jobs, nb_cores)) as p:
        res = p.map(_job, kwargs_list)

    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate DPGLM with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=25, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=100, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=100, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=1, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=2500, type=int)
    parser.add_argument('--svi_iters', help='SVI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=512, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=True)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=False)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    from sklearn.decomposition import PCA
    from scipy import stats

    from mimo.util.prediction import parallel_meanfield_prediction

    import tikzplotlib

    np.random.seed(args.seed)

    # sample dataset
    nb_samples = 10000
    input = np.linspace(-10., 10., nb_samples).reshape(nb_samples, 1)
    noise = lambda x: 0.05 + 0.2 * (1. + np.sin(2. * x)) / (1. + np.exp(-0.2 * x))
    target = np.sinc(input) + noise(input) * np.random.randn(len(input), 1)
    mean = np.sinc(input)

    data = np.hstack((input, target))

    # Data scaling
    input_scaler = PCA(n_components=1, whiten=True)
    target_scaler = PCA(n_components=1, whiten=True)

    input_scaler.fit(data[:, :1])
    target_scaler.fit(data[:, 1:])

    # shuffle data
    from sklearn.utils import shuffle
    data = shuffle(data)

    # split to nb_seeds train datasets
    from sklearn.model_selection import ShuffleSplit
    spliter = ShuffleSplit(n_splits=args.nb_seeds, test_size=0.2)

    train_inputs, train_targets = [], []
    for train_index, _ in spliter.split(data):
        train_inputs.append(data[train_index, :1])
        train_targets.append(data[train_index, 1:])

    train_data = []
    for train_input, train_target in zip(train_inputs, train_targets):
        train_data.append({'input': input_scaler.transform(train_input),
                           'target': target_scaler.transform(train_target)})

    # train
    dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                      train_data=train_data,
                                      arguments=args)

    # Evaluation over multiple seeds to get confidence
    mu_predict, std_predict,  = [], []
    evar, mse, smse = [], [], []
    for dpglm in dpglms:
        _mu_predict, _var_predict, _std_predict, _ =\
            parallel_meanfield_prediction(dpglm, input,
                                          prediction=args.prediction,
                                          input_scaler=input_scaler,
                                          target_scaler=target_scaler,
                                          sparse=True)

        _mse = mean_squared_error(target, _mu_predict)
        _evar = explained_variance_score(target, _mu_predict, multioutput='variance_weighted')
        _smse = 1. - r2_score(target, _mu_predict, multioutput='variance_weighted')

        print('EVAR:', _evar, 'MSE:', _mse, 'SMSE:', _smse,
              'Compnents:', len(dpglm.used_labels))

        mu_predict.append(_mu_predict)
        std_predict.append(_std_predict)
        evar.append(_evar)
        mse.append(_mse)
        smse.append(_smse)

    mu_predict = np.asarray(mu_predict).squeeze()
    std_predict = np.asarray(std_predict).squeeze()

    # calcule means and confidence intervals
    mu_predict_avg, mu_predict_std = np.mean(mu_predict, axis=0), np.std(mu_predict, axis=0)
    std_predict_avg, std_predict_std = np.mean(std_predict, axis=0), np.std(std_predict, axis=0)

    # plot mean and standard deviation of mean estimation
    w, h = plt.figaspect(0.67)
    fig, axes = plt.subplots(2, 1, figsize=(w, h))

    # plot data and prediction
    axes[0].plot(input, mean, 'k--', zorder=10)
    axes[0].scatter(input, target, s=0.75, facecolors='none', edgecolors='grey', zorder=1)
    axes[0].plot(input, mu_predict_avg, '-r', zorder=5)
    for c in [1., 2.]:
        axes[0].fill_between(input.flatten(),
                             mu_predict_avg - c * mu_predict_std,
                             mu_predict_avg + c * mu_predict_std,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1), zorder=1)

    # plot mean and standard deviation of data generation / estimated noise level
    axes[1].plot(input, noise(input), 'k--', zorder=10)
    axes[1].plot(input, std_predict_avg, '-r', zorder=5)
    for c in [1., 2.]:
        axes[1].fill_between(input.flatten(),
                             std_predict_avg - c * std_predict_std,
                             std_predict_avg + c * std_predict_std,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1), zorder=1)
    plt.tight_layout()

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'sinc'

    # save figs
    path = os.path.join(str(dataset) + '/')
    tikzplotlib.save(path + dataset + '_mean.tex')
    plt.savefig(path + dataset + '_mean.pdf')

    # plt.show()

    # show on learned example
    choice = np.random.choice(args.nb_seeds, 1)[0]
    w, h = plt.figaspect(0.67)
    fig, axes = plt.subplots(2, 1, figsize=(w, h))

    # plot data and prediction
    axes[0].plot(input, mean, 'k--', zorder=10)
    axes[0].plot(input, mean + 2 * noise(input), 'g--', zorder=10)
    axes[0].plot(input, mean - 2 * noise(input), 'g--', zorder=10)
    axes[0].scatter(input, target, s=0.75, facecolors='none', edgecolors='grey', zorder=1)

    axes[0].plot(input, mu_predict[choice], '-r', zorder=5)
    axes[0].plot(input, mu_predict[choice] + 2 * std_predict[choice], '-b', zorder=5)
    axes[0].plot(input, mu_predict[choice] - 2 * std_predict[choice], '-b', zorder=5)

    # plot gaussian activations
    mu_basis, sigma_basis = [], []
    for idx, c in enumerate(dpglms[choice].components):
        if idx in dpglms[choice].used_labels:
            _mu, _sigma, _, _ = c.posterior.mode()

            _mu = input_scaler.inverse_transform(np.atleast_2d(_mu))
            trans = (np.sqrt(input_scaler.explained_variance_[:, None]) * input_scaler.components_).T
            _sigma = trans.T @ np.diag(_sigma) @ trans

            mu_basis.append(_mu)
            sigma_basis.append(_sigma)

    activations = []
    for i in range(len(dpglms[choice].used_labels)):
        activations.append(stats.norm.pdf(input, mu_basis[i], np.sqrt(sigma_basis[i])))

    activations = np.asarray(activations).squeeze()
    activations = activations / np.sum(activations, axis=0, keepdims=True)

    for i in range(len(dpglms[choice].used_labels)):
        axes[1].plot(input, activations[i])

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'sinc'

    # save figs
    import tikzplotlib
    path = os.path.join(str(dataset) + '/')
    tikzplotlib.save(path + dataset + '_example.tex')
    plt.savefig(path + dataset + '_example.pdf')

    # plt.show()
