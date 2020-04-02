import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, mixture
from mimo.util.text import progprint_xrange

import argparse

import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate DPGLM with a Stick-breaking prior')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020/toy'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=5000, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=50, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=50, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=1000, type=int)
    parser.add_argument('--svi_iters', help='SVI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=256, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=False)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=False)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--nb_train', help='size of train dataset', default=500, type=int)
    parser.add_argument('--nb_splits', help='number of dataset splits', default=25, type=int)
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # create Chrip data
    from scipy.signal import chirp

    nb_train = args.nb_train

    x = np.linspace(0, 5, nb_train)[:, None]
    y = chirp(x, f0=2.5, f1=1., t1=2.5, method='hyperbolic') + 0.25 * npr.randn(nb_train, 1)
    data = np.hstack((x, y))

    input, target = data[:, :1], data[:, 1:]

    # scale data
    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=1, whiten=True)
    target_scaler = PCA(n_components=1, whiten=True)

    input_scaler.fit(data[:, :1])
    target_scaler.fit(data[:, 1:])

    # prepare model
    input_dim, target_dim = 1, 1

    nb_params = input_dim
    if args.affine:
        nb_params += 1

    components_prior = []

    # initialize Normal
    psi_niw = 5 * 1e-2
    kappa = 1e-2  # (1. / (input.T @ input)).item()

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

    # define prior
    if args.prior == 'stick-breaking':
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)),
                                deltas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = distributions.StickBreaking(**gating_hypparams)

        dpglm = mixture.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                                components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i])
                                            for i in range(args.nb_models)])

    elif args.prior == 'dirichlet':
        gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models,)) * args.alpha)
        gating_prior = distributions.Dirichlet(**gating_hypparams)

        dpglm = mixture.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                                components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i])
                                            for i in range(args.nb_models)])

    else:
        raise NotImplementedError

    anim = []

    split_size = int(nb_train / args.nb_splits)
    for n in range(args.nb_splits):
        print('Processing data split ' + str(n + 1) + ' out of ' + str(args.nb_splits))

        # # remove old data
        # try:
        #     dpglm.labels_list.pop()
        # except IndexError:
        #     print('Model has no data')

        _input = input[n * split_size: (n + 1) * split_size, :]
        _target = target[n * split_size: (n + 1) * split_size, :]

        # ada new data
        _input_scaled = input_scaler.transform(_input)
        _target_scaled = target_scaler.transform(_target)

        _data_scaled = np.hstack((_input_scaled, _target_scaled))

        dpglm.add_data(_data_scaled)

        # set posterior to prior
        import copy
        dpglm.gating.prior = copy.deepcopy(dpglm.gating.posterior)
        for i in range(len(dpglm.components)):
            dpglm.components[i].prior = copy.deepcopy(dpglm.components[i].posterior)

        # train model
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
                prob = batch_size / float(len(_data_scaled))
                for _ in svi_iter:
                    minibatch = npr.permutation(len(_data_scaled))[:batch_size]
                    dpglm.meanfield_sgdstep(minibatch=_data_scaled[minibatch, :],
                                            prob=prob, stepsize=args.svi_stepsize)
            if args.deterministic:
                # Meanfield VI
                if args.verbose:
                    print("Variational Inference")
                dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                                   maxiter=args.meanfield_iters,
                                                   progprint=args.verbose)

        # predict on all data
        from mimo.util.prediction import meanfield_prediction

        mu, var, std, nlpd = [], [], [], []
        for t in range(len(input)):
            _mean, _var, _std, _nlpd = meanfield_prediction(dpglm, input[t, :],
                                                            target[t, :],
                                                            prediction=args.prediction,
                                                            input_scaler=input_scaler,
                                                            target_scaler=target_scaler)

            mu.append(_mean)
            var.append(_var)
            std.append(_std)
            nlpd.append(_nlpd)

        mu = np.hstack(mu)
        var = np.hstack(var)
        std = np.hstack(std)
        nlpd = np.hstack(nlpd)

        # plot prediction
        fig = plt.figure(figsize=(12, 4))
        plt.scatter(input, target, s=0.75, color='k')
        plt.axvspan(_input.min(),  _input.max(), facecolor='grey', alpha=0.1)
        plt.plot(input, mu, color='crimson')

        for c in [1., 2.]:
            plt.fill_between(input.flatten(), mu - c * std, mu + c * std, color=(0, 0, 1, 0.05))

        plt.ylim((-2.5, 2.5))

        anim.append(fig)

        plt.show()
        plt.pause(1)

        # # set working directory
        # os.chdir(args.evalpath)
        # dataset = 'chirp'
        #
        # # save tikz and pdf
        # import tikzplotlib
        #
        # path = os.path.join(str(dataset) + '/')
        # tikzplotlib.save(path + dataset + '_' + str(n) + '.tex')
        # plt.savefig(path + dataset + '_' + str(n) + '.pdf')

    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage

    fps = 10
    def make_frame(t):
        idx = int(t * fps)
        return mplfig_to_npimage(anim[idx])

    # set working directory
    os.chdir(args.evalpath)
    dataset = 'chirp'
    path = os.path.join(str(dataset) + '/')

    animation = VideoClip(make_frame, duration=2.5)
    animation.write_gif(path + dataset + '.gif', fps=fps)
