import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, models

import os
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

    input_dim = input.shape[-1]
    target_dim = target.shape[-1]

    from sklearn.cluster import KMeans
    km = KMeans(args.nb_models).fit(np.hstack((input, target)))

    # set random seed
    np.random.seed(seed=seed)

    # initialize prior parameters: draw from uniform disitributions
    nb_params = input_dim
    if args.affine:
        nb_params += 1

    # initialize covariance scaler
    kappa = npr.uniform(0, 0.01)

    # initialize Matrix-Normal-Inverse-Wishart of output
    psi_mniw = npr.uniform(0, 0.1)

    V = np.diag(npr.uniform(0, 1e3, size=nb_params))
    # V[-1, -1] = npr.uniform(0, 1e3)  # higher variance for offset

    # define gating
    if args.prior == 'stick-breaking':
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)), deltas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = distributions.StickBreaking(**gating_hypparams)
    else:
        gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = distributions.Dirichlet(**gating_hypparams)

    components_prior = []
    for n in range(args.nb_models):
        _mu_input = km.cluster_centers_[n, :input_dim]
        _cov_input = np.array([np.cov(input[km.labels_ == n].T)])

        _mu_output = np.zeros((target_dim, nb_params))
        _mu_output[:, -1] = km.cluster_centers_[n, input_dim:]
        components_hypparams = dict(mu=_mu_input, kappa=kappa,
                                    psi_niw=_cov_input.reshape(input_dim, input_dim), nu_niw=input_dim + 1,
                                    M=_mu_output, V=V, affine=args.affine,
                                    psi_mniw=np.eye(target_dim) * psi_mniw, nu_mniw=target_dim + 1)

        aux = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)
        components_prior.append(aux)

    # define model
    if args.prior == 'stick-breaking':
        dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                               components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
    else:
        dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                               components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
    dpglm.add_data(np.hstack((input, target)))

    # Gibbs sampling to wander around the posterior
    for _ in range(args.gibbs_iters):
        dpglm.resample_model()

    # Mean field
    dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                       maxiter=args.meanfield_iters,
                                       progprint=False)

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
    parser.add_argument('--evalpath', help='Set path to dataset', default=os.path.abspath(mimo.__file__ + '/../../evaluation'))
    parser.add_argument('--nb_seeds', help='Set number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='Set prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='Set concentration parameter', default=100, type=float)
    parser.add_argument('--nb_models', help='Set max number of models', default=50, type=int)
    parser.add_argument('--affine', help='Set affine or not', default=True, type=bool)
    parser.add_argument('--gibbs_iters', help='Set Gibbs iterations', default=100, type=int)
    parser.add_argument('--meanfield_iters', help='Set max. VI iterations', default=500, type=int)
    parser.add_argument('--prediction', help='Set prediction to mode or average', default='average')
    parser.add_argument('--earlystop', help='Set stopping criterion for VI', default=1e-2, type=float)

    args = parser.parse_args()

    np.random.seed(1337)

    input = np.linspace(-10., 10., 1000)
    noise = lambda x: 0.05 + 0.2 * (1. + np.sin(2. * x)) / (1. + np.exp(-0.2 * x))
    target = np.sinc(input) + noise(input) * np.random.randn(len(input))
    mean = np.sinc(input)

    plt.plot(input, mean)
    plt.plot(input, mean + 2 * noise(input), '--g')
    plt.plot(input, mean - 2 * noise(input), '--g')
    plt.scatter(input, target, s=1, c='k')

    train_data = {'input': np.atleast_2d(input).T, 'target': np.atleast_2d(target).T}

    dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                      train_data=train_data,
                                      arguments=args)

    from mimo.util.prediction import meanfield_prediction

    mu_predic, std_predict = [], []
    for t in range(len(input)):
        _mu, _, _std = meanfield_prediction(dpglms[0], train_data['input'][t, :], prediction=args.prediction)
        mu_predic.append(_mu)
        std_predict.append(_std)

    mu_predic = np.vstack(mu_predic)
    std_predict = np.vstack(std_predict)

    plt.plot(train_data['input'], mu_predic, '-c')
    plt.plot(train_data['input'], mu_predic + 2 * std_predict, '-r')
    plt.plot(train_data['input'], mu_predic - 2 * std_predict, '-r')

    plt.figure()
    plt.plot(std_predict)
    plt.plot(noise(train_data['input']))
