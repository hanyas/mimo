import os
# os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, mixture
from mimo.util.text import progprint_xrange

import argparse

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
            dpglm.resample_model(nb_cores)

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
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=100, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=500, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=100, type=int)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=True)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=5000, type=int)
    parser.add_argument('--svi_iters', help='stochastic VI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='svi step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='svi batch size', default=1024, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=True)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--nb_train', help='size of train dataset', default=10000, type=int)
    parser.add_argument('--nb_test', help='size of test dataset', default=2000, type=int)
    parser.add_argument('--seed', help='choose seed', default=1, type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)

    import scipy as sc
    from scipy import io

    # load all available data
    data = sc.io.loadmat(args.datapath + '/Barrett/ias_real_barrett_data.mat')
    all_data = np.vstack((np.hstack((data['X_train'], data['Y_train'])),
                          np.hstack((data['X_test'], data['Y_test']))))
    # shuffle data
    from sklearn.utils import shuffle
    all_data = shuffle(all_data)

    # training data
    nb_train = args.nb_train
    train_data = all_data[:nb_train, :]
    train_input, train_target = train_data[:, :21], train_data[:, 21:]

    # test data
    nb_test = args.nb_test
    test_data = all_data[-nb_test:, :]
    test_input, test_target = test_data[:, :21], test_data[:, 21:]

    # scale training data
    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=21, whiten=True)
    target_scaler = PCA(n_components=7, whiten=True)

    input_scaler.fit(all_data[:, :21])
    target_scaler.fit(all_data[:, 21:])

    train_data = {'input': input_scaler.transform(train_input),
                  'target': target_scaler.transform(train_target)}

    # train
    dpglm = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                     train_data=train_data,
                                     arguments=args)[0]

    from mimo.util.prediction import parallel_meanfield_prediction
    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

    # # predict on train data
    # train_predict, _, _ = parallel_meanfield_prediction(dpglm, train_input,
    #                                                     prediction=args.prediction,
    #                                                     input_scaler=input_scaler,
    #                                                     target_scaler=target_scaler)
    #
    # train_evar = explained_variance_score(train_target, train_predict)
    # train_mse = mean_squared_error(train_target, train_predict)
    # train_smse = 1. - r2_score(train_target, train_predict)
    #
    # train_nlpd = - 1.0 * dpglm.predictive_log_likelihood(np.hstack((input_scaler.transform(train_input),
    #                                                                 target_scaler.transform(train_target)))).mean()
    #
    # print('TRAIN - EVAR:', train_evar, 'MSE:', train_mse,
    #       'SMSE:', train_smse, 'NLPD:', train_nlpd,
    #       'Compnents:', len(dpglm.used_labels))

    # predict on test data
    test_predict, _, _ = parallel_meanfield_prediction(dpglm, test_input,
                                                       prediction=args.prediction,
                                                       input_scaler=input_scaler,
                                                       target_scaler=target_scaler)

    test_evar = explained_variance_score(test_target, test_predict)
    test_mse = mean_squared_error(test_target, test_predict)
    test_smse = 1. - r2_score(test_target, test_predict)

    test_nlpd = - 1.0 * dpglm.predictive_log_likelihood(np.hstack((input_scaler.transform(test_input),
                                                                   target_scaler.transform(test_target)))).mean()

    print('TEST - EVAR:', test_evar, 'MSE:', test_mse,
          'SMSE:', test_smse, 'NLPD:', test_nlpd,
          'Compnents:', len(dpglm.used_labels))

    arr = np.hstack([test_evar, test_mse,
                     test_smse, test_nlpd,
                     len(dpglm.used_labels)])

    np.savetxt('barrett_' + str(args.prior) + '_' + str(args.nb_train) + '_alpha_' + str(args.alpha) + '_seed_' + str(args.seed) + '.csv', arr, delimiter=',')
