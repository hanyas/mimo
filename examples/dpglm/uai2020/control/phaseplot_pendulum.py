import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, mixture
from mimo.util.text import progprint_xrange
from mimo.util.plot import beautify

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
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020/control'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=10, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=25, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=100, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=1000, type=int)
    parser.add_argument('--svi_iters', help='stochastic VI iterations', default=1000, type=int)
    parser.add_argument('--svi_stepsize', help='svi step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='svi batch size', default=1024, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=True)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    import gym

    from mimo.util.data import sample_env

    np.random.seed(args.seed)

    env = gym.make('Pendulum-DPGLM-v1')
    env._max_episode_steps = 5000
    env.unwrapped._dt = 0.01
    env.unwrapped._sigma = 1e-4
    env.seed(args.seed)

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    nb_train_rollouts, nb_train_steps = 25, 250
    nb_test_rollouts, nb_test_steps = 5, 250

    train_obs, train_act = sample_env(env, nb_train_rollouts, nb_train_steps)

    train_input = np.vstack([np.hstack((_x[:-1, :], _u[:-1, :])) for _x, _u in zip(train_obs, train_act)])
    train_target = np.vstack([_x[1:, :] - _x[:-1, :] for _x in train_obs])

    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=dm_obs + dm_act, whiten=True)
    target_scaler = PCA(n_components=dm_obs, whiten=True)

    input_scaler.fit(train_input)
    target_scaler.fit(train_target)

    scaled_train_data = {'input': input_scaler.transform(train_input),
                         'target': target_scaler.transform(train_target)}

    dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                      train_data=scaled_train_data,
                                      arguments=args)

    # create meshgrid
    xlim = (-np.pi, np.pi)
    ylim = (-8.0, 8.0)

    npts = 26

    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)

    X, Y = np.meshgrid(x, y)
    XY = np.stack((X, Y))

    XYn = np.zeros((3, npts, npts))

    env.reset()
    for i in range(npts):
        for j in range(npts):
            XYn[:, i, j] = env.unwrapped.fake_step(XY[:, i, j], np.array([0.0]))

    Yn = XYn[2, :, :]

    # Transform to angle
    cos_x = XYn[0, :, :]
    sin_x = XYn[1, :, :]
    Xn = np.arctan2(sin_x, cos_x)

    XYn = np.stack((Xn, Yn))

    dydt = XYn - XY

    # streamplot for environment
    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    ax.streamplot(x, y, dydt[0, ...], dydt[1, ...],
                  color='b', linewidth=0.75, density=1.,
                  arrowstyle='->', arrowsize=1.5)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(xlim)
    ax.set_xlabel('Angle')

    ax.set_ylim(ylim)
    ax.set_ylabel('Velocity')

    plt.title('Pendulum - Phase Plot of True Dynamics')

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'pendulum'

    # save tikz and pdf
    import tikzplotlib

    path = os.path.join(str(dataset) + '/')
    tikzplotlib.save(path + dataset + '_phaseplot_env.tex')
    plt.savefig(path + dataset + '_phaseplot_env.pdf')

    plt.show()

    # streamplot for predicted next states
    from mimo.util.prediction import meanfield_prediction

    # transform to angle to cos / sin
    X1 = np.cos(X)
    X2 = np.sin(X)

    XY = np.stack((X1, X2, Y))

    # next states from environment
    XYn = np.zeros((3, npts, npts))
    for i in range(npts):
        for j in range(npts):
            _cos_x = XY[0, i, j]
            _sin_x = XY[1, i, j]
            _vel = XY[2, i, j]
            _action = 0
            test_obs = np.asarray([_cos_x, _sin_x, _vel, _action])

            prediction = meanfield_prediction(dpglms[0], test_obs, incremental=True,
                                              input_scaler=input_scaler,
                                              target_scaler=target_scaler)
            XYn[:, i, j] = prediction[0]

    # retransform to angle
    cos_x = XYn[0, :, :]
    sin_x = XYn[1, :, :]
    Xn = np.arctan2(sin_x, cos_x)
    Yn = XYn[2, :, :]

    XYn = np.stack((Xn, Yn))
    XY = np.stack((X, Y))

    dydt = XYn - XY

    # streamplot prediction
    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()
    ax.streamplot(x, y, dydt[0, ...], dydt[1, ...],
                  color='r', linewidth=0.75, density=1.,
                  arrowstyle='->', arrowsize=1.5)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(xlim)
    ax.set_xlabel('Angle')

    ax.set_ylim(ylim)
    ax.set_ylabel('Angular Velocity')

    plt.title('Pendulum - Phase Plot of Learned Dynamics')

    # save tikz and pdf
    import tikzplotlib

    path = os.path.join(str(dataset) + '/')
    tikzplotlib.save(path + dataset + '_phaseplot_learned.tex')
    plt.savefig(path + dataset + '_phaseplot_learned.pdf')

    plt.show()
