import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, mixture
from mimo.util.text import progprint_xrange

import argparse

import matplotlib.pyplot as plt

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
    psi_niw = 1e0
    kappa = 1e-2

    # initialize Matrix-Normal
    psi_mniw = 1e0
    V = 1e3 * np.eye(nb_params)

    for n in range(args.nb_models):
        basis_hypparams = dict(mu=np.zeros((input_dim, )),
                               psi=np.eye(input_dim) * psi_niw,
                               kappa=kappa, nu=input_dim + 1)

        aux = distributions.NormalInverseWishart(**basis_hypparams)
        basis_prior.append(aux)

        models_hypparams = dict(M=np.zeros((target_dim, nb_params)),
                                affine=args.affine, V=V,
                                nu=target_dim + 1,
                                psi=np.eye(target_dim) * psi_mniw)

        aux = distributions.MatrixNormalInverseWishart(**models_hypparams)
        models_prior.append(aux)

    # define gating
    if args.prior == 'stick-breaking':
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)), deltas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = distributions.StickBreaking(**gating_hypparams)

        dpglm = mixture.BayesianMixtureOfLinearGaussians(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                                                         basis=[distributions.BayesianGaussian(basis_prior[i]) for i in range(args.nb_models)],
                                                         models=[distributions.BayesianLinearGaussian(models_prior[i]) for i in range(args.nb_models)])

    else:
        gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = distributions.Dirichlet(**gating_hypparams)

        dpglm = mixture.BayesianMixtureOfLinearGaussians(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                                                         basis=[distributions.BayesianGaussian(basis_prior[i]) for i in range(args.nb_models)],
                                                         models=[distributions.BayesianLinearGaussian(models_prior[i]) for i in range(args.nb_models)])
    dpglm.add_data(target, input, whiten=False)

    for _ in range(args.super_iters):
        # Gibbs sampling
        if args.verbose:
            print("Gibbs Sampling")

        gibbs_iter = range(args.gibbs_iters) if not args.verbose\
            else progprint_xrange(args.gibbs_iters)

        for _ in gibbs_iter:
            dpglm.resample()

        if args.stochastic:
            # Stochastic meanfield VI
            if args.verbose:
                print('Stochastic Variational Inference')

            svi_iter = range(args.gibbs_iters) if not args.verbose\
                else progprint_xrange(args.svi_iters)

            batch_size = args.svi_batchsize
            prob = batch_size / float(len(input))
            for _ in svi_iter:
                minibatch = npr.permutation(len(input))[:batch_size]
                dpglm.meanfield_sgdstep(y=target[minibatch, :], x=input[minibatch, :],
                                        prob=prob, stepsize=args.svi_stepsize)
        if args.deterministic:
            # Meanfield VI
            if args.verbose:
                print("Variational Inference")
            dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                               maxiter=args.meanfield_iters,
                                               progprint=args.verbose)

        dpglm.gating.prior = dpglm.gating.posterior
        for i in range(dpglm.size):
            dpglm.basis[i].prior = dpglm.basis[i].posterior
            dpglm.models[i].prior = dpglm.models[i].posterior

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
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=1, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=5, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=False)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=3, type=int)
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

    dpglm = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                     train_input=trans_input,
                                     train_target=target,
                                     arguments=args)[0]

    # predict
    mu_predict, var_predict, std_predict = [], [], []
    for t in range(len(trans_input)):
        _mean, _var, _, _ = dpglm.meanfield_prediction(trans_input[t, :], prediction=args.prediction)

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

    fig, axes = plt.subplots(2, 1)

    # plot prediction
    sorter = np.argsort(input, axis=0).flatten()
    sorted_input, sorted_target = input[sorter, 0], target[sorter, 0]
    sorted_mu_predict, sorted_std_predict = mu_predict[sorter, 0], std_predict[sorter, 0]

    axes[0].scatter(sorted_input, sorted_target, s=0.75, color='k')
    axes[0].plot(sorted_input, sorted_mu_predict, color='crimson')
    for c in [1., 2.]:
        axes[0].fill_between(sorted_input,
                             sorted_mu_predict - c * sorted_std_predict,
                             sorted_mu_predict + c * sorted_std_predict,
                             edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1))

    axes[0].set_ylabel('y')

    # plot gaussian activations
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p(x)')

    activations = dpglm.meanfield_predictive_activation(sorted_input)
    for i in range(len(dpglm.used_labels)):
        axes[1].plot(sorted_input, activations[:, i])

    # set working directory
    dataset = 'step_poly_features'
    try:
        os.chdir(args.evalpath + '/' + dataset)
    except FileNotFoundError:
        os.makedirs(args.evalpath + '/' + dataset, exist_ok=True)
        os.chdir(args.evalpath + '/' + dataset)

    # save tikz and pdf
    import tikzplotlib
    tikzplotlib.save(dataset + '.tex')
    plt.savefig(dataset + '.pdf')

    plt.show()
