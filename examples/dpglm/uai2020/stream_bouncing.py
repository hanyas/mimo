import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, mixture
from mimo.util.text import progprint_xrange
from mimo.util.general import near_pd, beautify

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
        dpglm = mixture.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                               components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
    else:
        dpglm = mixture.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
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
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020/toy'))
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
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=500, type=int)
    parser.add_argument('--svi_iters', help='stochastic VI iterations', default=2500, type=int)
    parser.add_argument('--svi_stepsize', help='svi step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='svi batch size', default=1024, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=True)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')

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

    train_input = np.vstack([_x[:-1, :] for _x in train_obs])
    train_target = np.vstack([_x[1:, :] - _x[:-1, :] for _x in train_obs])

    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=dm_obs, whiten=True)
    target_scaler = PCA(n_components=dm_obs, whiten=True)

    input_scaler.fit(train_input)
    target_scaler.fit(train_target)

    scaled_train_data = {'input': input_scaler.transform(train_input),
                         'target': target_scaler.transform(train_target)}

    dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                      train_data=scaled_train_data,
                                      arguments=args)

    # create meshgrid
    xlim = (0.1, 5)
    ylim = (-8.0, 8.0)

    npts = 26

    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)

    X, Y = np.meshgrid(x, y)
    XY = np.stack((X, Y))

    # next states from environment
    XYn = np.zeros((2, npts, npts))
    Zn = np.zeros((npts, npts))

    env.reset()
    for i in range(npts):
        for j in range(npts):
            XYn[:, i, j] = env.unwrapped.fake_step(XY[:, i, j], np.array([0.0]))
            # Zn[i, j], XYn[:, i, j] = env.unwrapped.fake_step(XY[:, i, j], np.array([0.0]))

    dydt = XYn - XY

    # streamplot environment
    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    ax.streamplot(x, y, dydt[0, ...], dydt[1, ...],
                  color='b', linewidth=0.75, density=1,
                  arrowstyle='->', arrowsize=1.5)# minlength=0.5 to control minimum length of streams

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(xlim)
    ax.set_xlabel('height')

    ax.set_ylim(ylim)
    ax.set_ylabel('velocity')

    plt.title('bouncing ball - streamplot for 1-step rollout (environment)')

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'bouncing'

    # save tikz and pdf
    import tikzplotlib
    path = os.path.join(str(dataset) + '/')
    tikzplotlib.save(path + dataset + '_stream_env.tex')
    plt.savefig(path + dataset + '_stream_env.pdf')

    plt.show()


    # predicted next states
    from mimo.util.prediction import meanfield_prediction
    for i in range(npts):
        for j in range(npts):
            h = XY[0, i, j]
            h_dot = XY[1, i, j]
            test_obs = np.asarray([h, h_dot])
            prediction = meanfield_prediction(dpglms[0], test_obs, incremental=True,
                                           input_scaler=input_scaler,
                                           target_scaler=target_scaler)
            XYn[:, i, j] = prediction[0]

    dydt = XYn - XY

    # streamplot prediction
    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()
    ax.streamplot(x, y, dydt[0, ...], dydt[1, ...],
                  color='r', linewidth=0.75, density=1,
                  arrowstyle='->', arrowsize=1.5)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(xlim)
    ax.set_xlabel('height')

    ax.set_ylim(ylim)
    ax.set_ylabel('velocity')

    plt.title('bouncing ball - streamplot for 1-step prediction')

    # save tikz and pdf
    import tikzplotlib
    path = os.path.join(str(dataset) + '/')
    tikzplotlib.save(path + dataset + '_stream_pred.tex')
    plt.savefig(path + dataset + '_stream_pred.pdf')

    plt.show()



