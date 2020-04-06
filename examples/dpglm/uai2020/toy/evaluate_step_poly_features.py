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
            _psi_niw = np.cov(input[km.labels_ == n], bias=False, rowvar=False)
            psi_niw = 1e0  # np.diag(near_pd(np.atleast_2d(_psi_niw)))
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
        psi_niw = 1e0
        kappa = 1e-2

        # initialize Matrix-Normal
        psi_mniw = 1e0
        if args.affine:
            X = np.hstack((input, np.ones((len(input), 1))))
        else:
            X = input

        V = 10 * X.T @ X

        for n in range(args.nb_models):
            components_hypparams = dict(mu=np.zeros((input_dim, )),
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
        kwargs['seed'] = n
        kwargs_list.append(kwargs.copy())

    with Pool(processes=min(nb_jobs, nb_cores)) as p:
        res = p.map(_job, kwargs_list)

    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate DPGLM with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=1, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=5, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=1, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=1000, type=int)
    parser.add_argument('--svi_iters', help='SVI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=128, type=int)
    parser.add_argument('--prediction', help='prediction to mode or average', default='mode')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=False)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # create data
    nb_train = 1200

    input, mean = [], []

    input.append(np.linspace(-3., 0, int(nb_train / 3)).reshape(int(nb_train / 3), 1))
    input.append(np.linspace(0., 3., int(nb_train / 3)).reshape(int(nb_train / 3), 1))
    input.append(np.linspace(3., 6., int(nb_train / 3)).reshape(int(nb_train / 3), 1))

    mean.append(-2 * input[0] ** 3 + 2 * input[0])
    mean.append(-2 * (input[1] - 3) ** 3 + 2 * (input[1] - 3))
    mean.append(-2 * (input[2] - 6) ** 3 + 2 * (input[2] - 6))

    input, mean = np.vstack(input), np.vstack(mean)
    noise = 3.0 * npr.randn(nb_train).reshape(nb_train, 1)
    target = mean + noise

    # polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True)
    trans_input = poly.fit_transform(np.atleast_2d(input))

    train_data = {'input': trans_input, 'target': target}

    dpglm = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                     train_data=train_data,
                                     arguments=args)[0]

    # predict
    from mimo.util.prediction import meanfield_prediction

    mu_predict, var_predict, std_predict = [], [], []
    for t in range(len(trans_input)):
        _mean, _var, _, _ = meanfield_prediction(dpglm, trans_input[t, :], prediction=args.prediction)

        mu_predict.append(_mean)
        var_predict.append(_var)
        std_predict.append(np.sqrt(_var))

    mu_predict = np.vstack(mu_predict)
    var_predict = np.vstack(var_predict)
    std_predict = np.vstack(std_predict)

    # metrics
    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

    mse = mean_squared_error(target, mu_predict)
    evar = explained_variance_score(target, mu_predict, multioutput='variance_weighted')
    smse = 1. - r2_score(target, mu_predict, multioutput='variance_weighted')

    print('EVAR:', evar, 'MSE:', mse, 'SMSE:', smse, 'Compnents:', len(dpglm.used_labels))

    import scipy.stats as stats

    fig, axes = plt.subplots(2, 1)

    # plot prediction
    sorter = np.argsort(input, axis=0).flatten()
    input, target = input[sorter, 0], target[sorter, 0]
    mu_predict, std_predict = mu_predict[sorter, 0], std_predict[sorter, 0]

    axes[0].scatter(input, target, s=0.75, color='k')
    axes[0].plot(input, mu_predict, color='crimson')
    for c in [1., 2.]:
        axes[0].fill_between(input, mu_predict - c * std_predict,
                             mu_predict + c * std_predict,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1))

    axes[0].set_ylabel('y')

    # plot gaussian activations
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p(x)')

    mu_basis, sigma_basis = [], []
    for idx, c in enumerate(dpglm.components):
        if idx in dpglm.used_labels:
            _mu, _sigma, _, _ = c.posterior.mode()

            mu_basis.append(_mu)
            sigma_basis.append(_sigma)

    activations = []
    for i in range(len(dpglm.used_labels)):
        activations.append(stats.norm.pdf(trans_input[:, 1], mu_basis[i][1], np.sqrt(sigma_basis[i][1, 1])))

    activations = np.asarray(activations).squeeze()
    activations = activations / np.sum(activations, axis=0, keepdims=True)

    for i in range(len(dpglm.used_labels)):
        axes[1].plot(trans_input[:, 1], activations[i])

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'step_poly_features'

    # save tikz and pdf
    import tikzplotlib
    path = os.path.join(str(dataset) + '/')
    tikzplotlib.save(path + dataset + '.tex')
    plt.savefig(path + dataset + '.pdf')

    plt.show()
