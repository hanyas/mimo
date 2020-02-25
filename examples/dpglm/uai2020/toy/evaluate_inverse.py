import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, mixture
from mimo.util.text import progprint_xrange

import os
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
            psi_niw = 1e0
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
        psi_mniw = 1e-1
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
    parser.add_argument('--datapath', help='Set path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='Set path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020/toy'))
    parser.add_argument('--nb_seeds', help='Set number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='Set prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='Set concentration parameter', default=25, type=float)
    parser.add_argument('--nb_models', help='Set max number of models', default=50, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--super_iters', help='Set interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Set Gibbs iterations', default=1, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='Set max VI iterations', default=1000, type=int)
    parser.add_argument('--svi_iters', help='Set stochastic VI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='Set SVI step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='Set SVI batch size', default=256, type=int)
    parser.add_argument('--prediction', help='Set prediction to mode or average', default='mode')
    parser.add_argument('--earlystop', help='Set stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=False)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # create data
    noise = npr.normal(0, 1, (200, 1)) * 0.05
    target = npr.uniform(0, 1, (200, 1))
    input = target + 0.3 * np.sin(2. * np.pi * target) + noise

    # creat plot for mean vs mode prediction and gaussian activations
    fig, axes = plt.subplots(2, 1)

    axes[0].scatter(input, target, facecolors='none',
                    edgecolors='k', linewidth=0.5)
    plt.ylabel('y')

    train_data = {'input': input, 'target': target}

    dpglm = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                     train_data=train_data,
                                     arguments=args)[0]

    # mean prediction
    from mimo.util.prediction import meanfield_prediction

    mu_predict = []
    for t in range(len(input)):
        _mean, _, _ = meanfield_prediction(dpglm, input[t, :], prediction='average')
        mu_predict.append(np.atleast_2d(_mean))

    mu_predict = np.vstack(mu_predict)

    # metrics
    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
    evar = explained_variance_score(target, mu_predict)
    mse = mean_squared_error(target, mu_predict)
    smse = 1. - r2_score(target, mu_predict)

    print('MEAN - EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'Components:', len(dpglm.used_labels))

    axes[0].scatter(input, mu_predict, marker='x', c='b', linewidth=0.5)

    # mode prediction
    from mimo.util.prediction import meanfield_prediction

    mu_predict = []
    for t in range(len(input)):
        _mean, _var, _ = meanfield_prediction(dpglm, input[t, :], prediction='mode')
        mu_predict.append(np.atleast_2d(_mean))

    mu_predict = np.vstack(mu_predict)
    # metrics
    from sklearn.metrics import explained_variance_score, mean_squared_error

    evar = explained_variance_score(target, mu_predict)
    mse = mean_squared_error(target, mu_predict)
    smse = 1. - r2_score(target, mu_predict)

    print('Mode - EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'Components:', len(dpglm.used_labels))

    axes[0].scatter(input, mu_predict, marker='D', facecolors='none', edgecolors='r', linewidth=0.5)

    # plot gaussian activations
    import scipy.stats as stats
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p(x)')

    mu, sigma = [], []
    for idx, c in enumerate(dpglm.components):
        if idx in dpglm.used_labels:
            _mu, _sigma, _, _ = c.posterior.mode()
            mu.append(_mu)
            sigma.append(_sigma)

    sorting = np.argsort(input, axis=0)  # sort based on input values for plotting
    sorted_input = np.take_along_axis(input, sorting, axis=0)
    activations = []
    for i in range(len(dpglm.used_labels)):
        activations.append(stats.norm.pdf(sorted_input, mu[i], np.sqrt(sigma[i])))

    activations = np.asarray(activations).squeeze()
    # activations = activations / np.sum(activations, axis=1, keepdims=True)
    activations = activations / np.sum(activations, axis=0, keepdims=True)

    colours = ['green', 'orange', 'purple']
    for i in range(len(dpglm.used_labels)):
        axes[1].plot(sorted_input, activations[i], color=colours[i])

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'inverse'

    # save tikz and pdf
    import tikzplotlib
    path = os.path.join(str(dataset) + '/')
    tikzplotlib.save(path + dataset + '_comparison.tex')
    plt.savefig(path + dataset + '_comparison.pdf')
    plt.show()

    # get mean of matrix-normal for plotting experts
    regcoeff = []
    for idx, c in enumerate(dpglm.components):
        if idx in dpglm.used_labels:
            _, _, _, _, M, _, _, _ = c.posterior.params
            regcoeff.append(M)

    # plot three experts
    plt.figure()
    axis = np.linspace(0, 1, 500).reshape(-1, 1)
    mu_predict = []
    for t in range(len(axis)):
        q = np.hstack((axis[t, :], 1.))
        _mu_predict = (regcoeff[0] @ q).tolist()
        mu_predict.append(_mu_predict )
    mu_predict = np.asarray(mu_predict).reshape(-1, 1)
    plt.plot(axis, mu_predict, linewidth=2, c='green')

    mu_predict = []
    for t in range(len(axis)):
        q = np.hstack((axis[t, :], 1.))
        _mu_predict = (regcoeff[1] @ q).tolist()
        mu_predict.append(_mu_predict )
    mu_predict = np.asarray(mu_predict).reshape(-1, 1)
    plt.plot(axis, mu_predict, linewidth=2, c='orange')

    mu_predict = []
    for t in range(len(axis)):
        q = np.hstack((axis[t, :], 1.))
        _mu_predict = (regcoeff[2] @ q).tolist()
        mu_predict.append(_mu_predict)
    mu_predict = np.asarray(mu_predict).reshape(-1, 1)
    plt.plot(axis, mu_predict, linewidth=2, c='purple')

    # plot data
    plt.scatter(input, target, facecolors='none', edgecolors='k', linewidth=0.5)

    plt.ylabel('y')
    plt.xlabel('x')

    # save tikz and pdf
    import tikzplotlib
    path = os.path.join(str(dataset) + '/')
    tikzplotlib.save(path + dataset + '_experts.tex')
    plt.savefig(path + dataset + '_experts.pdf')

    plt.show()
