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

    # set random seed
    np.random.seed(seed=None)

    # initialize prior parameters: draw from uniform disitributions
    nb_params = input_dim
    if args.affine:
        nb_params += 1

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

    V = np.diag(npr.uniform(0, 1e1, size=nb_params))
    # V[-1, -1] = npr.uniform(0, 1e1)  # higher variance for offset

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
    parser.add_argument('--prediction', help='Set prediction to mode or average', default='mode')
    parser.add_argument('--earlystop', help='Set stopping criterion for VI', default=1e-2, type=float)

    args = parser.parse_args()

    import gym

    from mimo.util.data import sample_env

    np.random.seed(1337)

    env = gym.make('BouncingBall-DPGLM-v0')
    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.01
    env.unwrapped._sigma = 1e-16
    env.seed(1337)

    dm_obs = env.observation_space.shape[0]

    nb_train_rollouts, nb_train_steps = 10, 500
    nb_test_rollouts, nb_test_steps = 5, 500

    train_obs, _ = sample_env(env, nb_train_rollouts, nb_train_steps)
    test_obs, _ = sample_env(env, nb_test_rollouts, nb_test_steps)

    _train_input = np.vstack([_x[:-1, :] for _x in train_obs])
    _train_target = np.vstack([_x[1:, :] - _x[:-1, :] for _x in train_obs])

    train_data = {'input': _train_input,
                  'target': _train_target}

    dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                      train_data=train_data,
                                      arguments=args)

    from mimo.util.prediction import meanfield_forcast

    _prediction = meanfield_forcast(dpglms[0], test_obs[0][0, :],
                                    horizon=500, incremental=True)

    plt.figure()
    plt.plot(test_obs[0])
    plt.plot(_prediction)

    # from sklearn.decomposition import PCA
    # _input_scaler = PCA(n_components=dm_obs, whiten=True)
    # _output_scaler = PCA(n_components=dm_obs, whiten=True)
    #
    # _input_scaler.fit(_train_input)
    # _output_scaler.fit(_train_target)
    #
    # scaled_train_data = {'input': _input_scaler.transform(_train_input),
    #                      'target': _output_scaler.transform(_train_target)}
    #
    # dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
    #                                   train_data=scaled_train_data,
    #                                   arguments=args)
    #
    # from mimo.util.prediction import scaled_meanfield_forcast
    #
    # _unscaled_prediction = scaled_meanfield_forcast(dpglms[0], test_obs[0][0, :],
    #                                                 horizon=500, incremental=True,
    #                                                 input_scaler=_input_scaler,
    #                                                 output_scaler=_output_scaler)
    #
    # plt.figure()
    # plt.plot(test_obs[0])
    # plt.plot(_unscaled_prediction)
