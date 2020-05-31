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
            _psi_niw = np.cov(train_data['input'][km.labels_ == n], bias=False, rowvar=False)
            psi_niw = np.diag(near_pd(np.atleast_2d(_psi_niw)))
            kappa = 1e-2

            # initialize Matrix-Normal
            mu_output = np.zeros((target_dim, nb_params))
            mu_output[:, -1] = km.cluster_centers_[n, input_dim:]
            psi_mniw = 1e0
            if args.affine:
                X = np.hstack((input, np.ones((len(input), 1))))
            else:
                X = input

            V = X.T @ X

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
        psi_niw = 1e-2
        kappa = 1e-2

        # initialize Matrix-Normal
        psi_mniw = 1e-1
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
                                        nu_mniw=target_dim + 23,
                                        psi_mniw=np.eye(target_dim) * psi_mniw)

            aux = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)
            components_prior.append(aux)

    # define gating
    if args.prior == 'stick-breaking':
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models,)),
                                deltas=np.ones((args.nb_models,)) * args.alpha)
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
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation'))
    parser.add_argument('--nb_seeds', help='number of seeds', default=5, type=int)
    parser.add_argument('--prior', help='prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='concentration parameter', default=2500, type=float)
    parser.add_argument('--nb_models', help='max number of models', default=5000, type=int)
    parser.add_argument('--affine', help='affine functions', action='store_true', default=True)
    parser.add_argument('--no_affine', help='non-affine functions', dest='affine', action='store_false')
    parser.add_argument('--super_iters', help='interleaving Gibbs/VI iterations', default=1, type=int)
    parser.add_argument('--gibbs_iters', help='Gibbs iterations', default=1, type=int)
    parser.add_argument('--deterministic', help='use deterministic VI', action='store_true', default=True)
    parser.add_argument('--no_deterministic', help='do not use deterministic VI', dest='deterministic', action='store_false')
    parser.add_argument('--meanfield_iters', help='max VI iterations', default=5000, type=int)
    parser.add_argument('--earlystop', help='stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--stochastic', help='use stochastic VI', action='store_true', default=False)
    parser.add_argument('--no_stochastic', help='do not use stochastic VI', dest='stochastic', action='store_false')
    parser.add_argument('--svi_iters', help='SVI iterations', default=500, type=int)
    parser.add_argument('--svi_stepsize', help='SVI step size', default=5e-4, type=float)
    parser.add_argument('--svi_batchsize', help='SVI batch size', default=1024, type=int)
    parser.add_argument('--prediction', help='prediction w/ mode or average', default='average')
    parser.add_argument('--kmeans', help='init with KMEANS', action='store_true', default=False)
    parser.add_argument('--no_kmeans', help='do not use KMEANS', dest='kmeans', action='store_false')
    parser.add_argument('--verbose', help='show learning progress', action='store_true', default=True)
    parser.add_argument('--mute', help='show no output', dest='verbose', action='store_false')
    parser.add_argument('--seed', help='choose seed', default=1337, type=int)

    args = parser.parse_args()

    import json
    print(json.dumps(vars(args), indent=4))

    np.random.seed(args.seed)

    import scipy as sc
    from scipy import io

    # load all available data
    _train_data = sc.io.loadmat(args.datapath + '/Sarcos/sarcos_inv.mat')['sarcos_inv']
    _test_data = sc.io.loadmat(args.datapath + '/Sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

    data = np.vstack((_train_data, _test_data))

    # scale data
    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=21, whiten=True)
    target_scaler = PCA(n_components=1, whiten=True)

    input_scaler.fit(data[:, :21])
    target_scaler.fit(data[:, 21:22])

    train_data = {'input': input_scaler.transform(_train_data[:, :21]),
                  'target': target_scaler.transform(_train_data[:, 21:22])}

    test_data = {'input': _test_data[:, :21],
                 'target': _test_data[:, 21:22]}

    # train
    dpglms = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                      train_data=train_data,
                                      arguments=args)

    from mimo.util.prediction import parallel_meanfield_prediction
    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

    test_evar, test_mse, test_smse, test_nlpd, nb_models = [], [], [], [], []
    for dpglm in dpglms:
        _nb_models = len(dpglm.used_labels)
        dpglm.clear_data()

        _test_predict, _, _, _test_nlpd =\
            parallel_meanfield_prediction(dpglm, test_data['input'],
                                          target=test_data['target'],
                                          prediction=args.prediction,
                                          input_scaler=input_scaler,
                                          target_scaler=target_scaler)

        _test_mse = mean_squared_error(test_data['target'], _test_predict)
        _test_evar = explained_variance_score(test_data['target'], _test_predict, multioutput='variance_weighted')
        _test_smse = 1. - r2_score(test_data['target'], _test_predict, multioutput='variance_weighted')

        print('TEST - EVAR:', _test_evar, 'MSE:', _test_mse,
              'SMSE:', _test_smse, 'NLPD:', _test_nlpd.mean(),
              'Compnents:', _nb_models)

        test_evar.append(_test_evar)
        test_mse.append(_test_mse)
        test_smse.append(_test_smse)
        test_nlpd.append(_test_nlpd.mean())
        nb_models.append(_nb_models)

    mean_mse = np.mean(test_mse)
    std_mse = np.std(test_mse)

    mean_smse = np.mean(test_smse)
    std_smse = np.std(test_smse)

    mean_evar = np.mean(test_evar)
    std_evar = np.std(test_evar)

    mean_nb_models = np.mean(nb_models)
    std_nb_models = np.std(nb_models)

    mean_nlpd = np.mean(test_nlpd)
    std_nlpd = np.std(test_nlpd)

    arr = np.array([mean_mse, std_mse,
                    mean_smse, std_smse,
                    mean_evar, std_evar,
                    mean_nb_models, std_nb_models,
                    mean_nlpd, std_nlpd])

    np.savetxt('sarcos_' + str(args.prior) +
               '_alpha_' + str(args.alpha) +
               '.csv', arr, delimiter=',')
