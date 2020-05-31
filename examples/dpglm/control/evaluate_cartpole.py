import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, mixture
from mimo.util.text import progprint_xrange
from mimo.util.general import near_pd

import argparse

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
        psi_niw = 1e0
        kappa = 1e-2

        # initialize Matrix-Normal
        psi_mniw = 1e0
        V = 1e3 * np.eye(nb_params)

        for n in range(args.nb_models):
            components_hypparams = dict(mu=np.zeros((input_dim, )),
                                        kappa=kappa, psi_niw=np.eye(input_dim) * psi_niw,
                                        nu_niw=input_dim + 1 + 100,
                                        M=np.zeros((target_dim, nb_params)),
                                        affine=args.affine, V=V,
                                        nu_mniw=target_dim + 1 + 100,
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
        dpglm = mixture.BayesianMixtureOfGaussians(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                                                   components=[distributions.BayesianJointLinearGaussian(components_prior[i])
                                                               for i in range(args.nb_models)])
    else:
        dpglm = mixture.BayesianMixtureOfGaussians(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                                                   components=[distributions.BayesianJointLinearGaussian(components_prior[i])
                                                               for i in range(args.nb_models)])
    dpglm.add_data(data)

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
            prob = batch_size / float(len(data))
            for _ in svi_iter:
                minibatch = npr.permutation(len(data))[:batch_size]
                dpglm.meanfield_sgdstep(obs=data[minibatch, :],
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
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/control'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=25, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=50, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=250, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=1, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=5000, type=int)
    parser.add_argument('--svi_iters', help='stochastic VI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='svi step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='svi batch size', default=512, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=False)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=False)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)
    parser.add_argument('--horizon', help='horizon prediction', default=1, type=int)
    parser.add_argument('--task', help='add task name suffix', default='')

    args = parser.parse_args()

    import gym

    from mimo.util.general import sample_env

    np.random.seed(args.seed)

    env = gym.make('Cartpole-DPGLM-v1')
    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.01
    env.unwrapped._sigma = 1e-2
    env.seed(args.seed)

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    nb_train_rollouts, nb_train_steps = 25, 250
    nb_test_rollouts, nb_test_steps = 5, 250

    train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)
    test_obs, test_act = sample_env(env, nb_test_rollouts, nb_test_steps)

    _input = np.vstack([np.hstack((_x[:-1, :], _u[:-1, :])) for _x, _u in zip(train_obs, train_act)])
    _target = np.vstack([_x[1:, :] - _x[:-1, :] for _x in train_obs])

    _train_data = np.hstack((_input, _target))

    # shuffle data
    from sklearn.utils import shuffle
    _train_data = shuffle(_train_data)

    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=dm_obs + dm_act, whiten=True)
    target_scaler = PCA(n_components=dm_obs, whiten=True)

    input_scaler.fit(_train_data[:, :dm_obs + dm_act])
    target_scaler.fit(_train_data[:, dm_obs + dm_act:])

    # split to nb_seeds train datasets
    from sklearn.model_selection import ShuffleSplit
    spliter = ShuffleSplit(n_splits=args.nb_seeds, test_size=0.2)

    train_inputs, train_targets = [], []
    for train_index, _ in spliter.split(_train_data):
        train_inputs.append(_train_data[train_index, :dm_obs + dm_act])
        train_targets.append(_train_data[train_index, dm_obs + dm_act:])

    train_data = []
    for train_input, train_target in zip(train_inputs, train_targets):
        train_data.append({'input': input_scaler.transform(train_input),
                           'target': target_scaler.transform(train_target)})

    # train
    dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                      train_data=train_data,
                                      arguments=args)

    # from mimo.util.prediction import meanfield_forcast
    #
    # idx = np.random.choice(len(test_obs))
    # prediction = meanfield_forcast(dpglms[0], test_obs[idx][0, :],
    #                                exogenous=test_act[idx],
    #                                horizon=len(test_act[idx]),
    #                                incremental=True,
    #                                input_scaler=input_scaler,
    #                                target_scaler=target_scaler)
    #
    # plt.figure()
    # plt.plot(test_obs[idx][1:, :])
    # plt.plot(prediction)
    #
    # plt.show()

    from mimo.util.prediction import kstep_error, parallel_meanfield_prediction

    test_input = np.vstack([np.hstack((_x[:-1, :], _u[:-1, :])) for _x, _u in zip(test_obs, test_act)])
    test_target = np.vstack([_x[1:, :] - _x[:-1, :] for _x in test_obs])

    mse, smse, evar, nb_models, duration, nlpd = [], [], [], [], [], []
    for dpglm in dpglms:
        _nb_models = len(dpglm.used_labels)
        _mse, _smse, _evar, _dur =\
            kstep_error(dpglm, test_obs, test_act, horizon=args.horizon,
                        input_scaler=input_scaler, target_scaler=target_scaler)

        _nlpd = parallel_meanfield_prediction(dpglm, test_input, target=test_target,
                                              input_scaler=input_scaler, target_scaler=target_scaler)[3]

        mse.append(_mse)
        smse.append(_smse)
        evar.append(_evar)
        nb_models.append(_nb_models)
        duration.append(_dur)
        nlpd.append(_nlpd.mean())

    mean_mse = np.mean(mse)
    std_mse = np.std(mse)

    mean_smse = np.mean(smse)
    std_smse = np.std(smse)

    mean_evar = np.mean(evar)
    std_evar = np.std(evar)

    mean_nb_models = np.mean(nb_models)
    std_nb_models = np.std(nb_models)

    mean_duration = np.mean(duration)
    std_duration = np.std(duration)

    mean_nlpd = np.mean(nlpd)
    std_nlpd = np.std(nlpd)

    arr = np.array([mean_mse, std_mse,
                    mean_smse, std_smse,
                    mean_evar, std_evar,
                    mean_nb_models, std_nb_models,
                    mean_duration, std_duration,
                    mean_nlpd, std_nlpd])

    if str(args.task) == 'alpha':
        np.savetxt('cartpole_' + str(args.prior) + '_' + str(args.task) + '_' + str(args.alpha) + '.csv', arr, delimiter=',')
    elif str(args.task) == 'models':
        np.savetxt('cartpole_' + str(args.prior) + '_' + str(args.task) + '_' + str(args.nb_models) + '.csv', arr, delimiter=',')
    elif str(args.task) == 'horizon':
        np.savetxt('cartpole_' + str(args.prior) + '_' + str(args.task) + '_' + str(args.horizon) + '.csv', arr, delimiter=',')
    else:
        raise NotImplementedError
