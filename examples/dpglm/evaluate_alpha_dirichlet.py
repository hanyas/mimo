import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, models
from mimo.util.data import load_data
from mimo.util.prediction import niw_marginal_likelihood, matrix_t
from mimo.util.prediction import sample_prediction
from mimo.util.plot import plot_violin_box

import os
import operator
import copy
import timeit
import datetime
import argparse

from joblib import Parallel, delayed

from sklearn.metrics import explained_variance_score
import csv
import pandas

import multiprocessing
nb_cores = multiprocessing.cpu_count()


def create_job(kwargs):
    train_data = kwargs.pop('train_data')
    test_data = kwargs.pop('test_data')
    input_dim = kwargs.pop('input_dim')
    output_dim = kwargs.pop('output_dim')
    args = kwargs.pop('arguments')

    # set random seed
    np.random.seed(seed=None)

    # initialize prior parameters: draw from uniform disitributions
    n_params = input_dim
    if args.affine:
        n_params = n_params + 1

    # initilaize Normal-Inverse-Wishart of input
    mu_low = np.amin(train_data[:, :-output_dim])
    mu_high = np.amax(train_data[:, :-output_dim])

    psi_niw = npr.uniform(0, 10)
    kappa = npr.uniform(0, 0.1)

    # initilaize Matrix-Normal-Inverse-Wishart of output
    psi_mniw = npr.uniform(0, 0.1)

    V = np.eye(n_params) * np.diag(npr.uniform(0, 10, size=n_params))
    V[-1, -1] = npr.uniform(0, 1000)  # higher variance for offset

    components_prior = []
    for _ in range(args.nb_models):
        components_hypparams = dict(mu=npr.uniform(mu_low, mu_high, size=input_dim),
                                    kappa=kappa, psi_niw=np.eye(input_dim) * psi_niw, nu_niw=input_dim + 1,
                                    M=np.zeros((output_dim, n_params)), V=V, affine=args.affine,
                                    psi_mniw=np.eye(output_dim) * psi_mniw, nu_mniw=output_dim * n_params + 1)

        aux = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)
        components_prior.append(aux)

    # define model
    if args.prior == 'stick-breaking':
        dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                               components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
    else:
        dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                               components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
    dpglm.add_data(train_data)

    # inference
    score = []

    # Gibbs sampling to wander around the posterior
    for _ in range(args.gibbs_iters):
        dpglm.resample_model()

    # Mean field
    score.append(dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                                    maxiter=args.meanfield_iters,
                                                    progprint=False))

    # # sample prediction
    # sample_prediction(dpglm, train_data)

    # compute posterior mixing weights
    weights = None
    if args.prior == 'dirichlet':
        weights = dpglm.gating.posterior.alphas
    elif args.prior == 'stick-breaking':
        product = np.ones((args.nb_models, ))
        gammas = dpglm.gating.posterior.gammas
        deltas = dpglm.gating.posterior.deltas
        for idx, c in enumerate(dpglm.components):
            product[idx] = gammas[idx] / (gammas[idx] + deltas[idx])
            for j in range(idx):
                product[idx] = product[idx] * (1. - gammas[j] / (gammas[j] + deltas[j]))
        weights = product

    # initialize variables
    mean, var, = np.zeros((n_test, output_dim)), np.zeros((n_test, output_dim)),

    mlklhd = np.zeros((args.nb_models, ))
    stats = []
    # get statistics for each component for training data
    for idx, c in enumerate(dpglm.components):
        _, _, _, _, _yxT, _xxT, _yyT, _n = c.posterior.get_statistics([l.data[l.z == idx] for l in dpglm.labels_list])
        stats.append({'xxT': _xxT, 'yxT': _yxT, 'yyT': _yyT, 'n': _n})

    # prediction / mean function of yhat for all training data xhat
    for i in range(n_test):
        xhat = test_data[i, :input_dim]

        # calculate the marginal likelihood of training data xhat for each cluster
        # calculate the normalization term for mean function for xhat
        normalizer = 0.
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                mlklhd[idx] = niw_marginal_likelihood(xhat, c.posterior)
                normalizer = normalizer + weights[idx] * mlklhd[idx]

        # calculate contribution of each cluster to mean function
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                t_mean, t_var, _ = matrix_t(xhat, c.prior, c.posterior, stats[idx])
                t_var = np.diag(t_var)  # consider only diagonal variances for plots

                # Mean of a mixture = sum of weihted means
                mean[i, :] += t_mean * mlklhd[idx] * weights[idx] / normalizer

                # Variance of a mixture = sum of weighted variances + ...
                # ... + sum of weighted squared means - squared sum of weighted means
                var[i, :] += (t_var + t_mean ** 2) * mlklhd[idx] * weights[idx] / normalizer
        var[i, :] -= mean[i, :] ** 2

    # # demo plots for CMB and Sine datasets
    # sorting = np.argsort(test_data, axis=0)  # sort based on input values
    # sorted_data = np.take_along_axis(test_data, sorting, axis=0)
    # sorted_mean = np.take_along_axis(mean, sorting[:, [0]], axis=0)
    # sorted_var = np.take_along_axis(var, sorting[:, [0]], axis=0)
    #
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(16, 6))
    # plt.scatter(test_data[:, 0], test_data[:, 1], s=1)
    # plt.plot(sorted_data[:, 0], sorted_mean, color='red')
    # plt.plot(sorted_data[:, 0], sorted_mean + 2. * np.sqrt(sorted_var), color='green')
    # plt.plot(sorted_data[:, 0], sorted_mean - 2. * np.sqrt(sorted_var), color='green')
    # plt.show()

    test_evar = explained_variance_score(test_data[:, input_dim:], mean, multioutput='variance_weighted')

    return dpglm, score, test_evar


def parallel_dpglm_inference(nb_jobs=50, **kwargs):
    kwargs_list = [kwargs for _ in range(nb_jobs)]
    results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
    dpglms, scores, test_evars = list(map(list, zip(*results)))
    return dpglms, scores, test_evars


if __name__ == "__main__":

    # timer
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Evaluate DPGLM with a Dirichlet prior')
    parser.add_argument('--dataset', help='Choose dataset', default='cmb')
    parser.add_argument('--datapath', help='Set path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='Set path to dataset', default=os.path.abspath(mimo.__file__ + '/../../evaluation'))
    parser.add_argument('--nb_seeds', help='Set number of seeds', default=100)
    parser.add_argument('--prior', help='Set prior type', default='dirichlet')
    parser.add_argument('--nb_models', help='Set max number of models', default=50)
    parser.add_argument('--affine', help='Set affine or not', default=True)
    parser.add_argument('--gibbs_iters', help='Set Gibbs iterations', default=1000)
    parser.add_argument('--meanfield_iters', help='Set VI iterations', default=500)
    parser.add_argument('--earlystop', help='Set stopping criterion for VI', default=1e-2)

    args = parser.parse_args()

    # prepare dataset
    data_file = args.dataset + '.csv'  # name of the file within datasets/
    if args.dataset == 'cmb':
        n_train = 600
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'sine':
        n_train = 1500
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'fk_1joint':
        n_train = 1000
        input_dim = 1
        output_dim = 2
    elif args.dataset == 'fk_2joint':
        n_train = 1000
        input_dim = 2
        output_dim = 2
    elif args.dataset == 'fk_3joint':
        n_train = 2500
        input_dim = 3
        output_dim = 2
    elif args.dataset == 'sarcos':
        n_train = 2500
        input_dim = 21
        output_dim = 7
    else:
        raise RuntimeError("Dataset does not exist")

    # concentration parameter of the prior
    alphas = []
    if args.prior == 'dirichlet':
        alphas = [0.01, 0.1, 1., 5., 10., 50., 100.]
    elif args.prior == 'stick-breaking':
        alphas = [0.1, 1., 10., 50., 100., 500., 1000.]

    # load dataset
    n_test = int(n_train / 5)
    train_data, test_data = load_data(n_train, n_test,
                                      data_file, args.datapath,
                                      output_dim, input_dim,
                                      args.dataset == 'sarcos',
                                      seed=1337)

    # save time stamp for file names
    time = str(datetime.datetime.now().strftime('_%m-%d_%H-%M-%S'))

    # set working directory and file name
    os.chdir(args.evalpath)
    tmp = 'alpha_' + str(args.prior)

    violin_data = None

    for itr, alpha in enumerate(alphas):
        print('Current alpha value', alpha)

        raw_path = os.path.join(str(args.dataset) + '/raw/' + str(args.dataset) + '_' + str(tmp) + '_' + str(alpha) + '_raw' + time + '.csv')
        stats_path = os.path.join(str(args.dataset) + '/stats/' + str(args.dataset) + '_' + str(tmp) + '_' + str(alpha) + '_stats' + time + '.csv')
        visual_tikz_path = os.path.join(str(args.dataset) + '/visual/' + str(args.dataset) + '_' + str(tmp))
        visual_pdf_path = os.path.join(str(args.dataset) + '/visual/' + str(args.dataset) + '_' + str(tmp))

        # define gating
        if args.prior == 'stick-breaking':
            gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models, )), deltas=np.ones((args.nb_models, )) * alpha)
            gating_prior = distributions.StickBreaking(**gating_hypparams)
        else:
            gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models, )) * alpha)
            gating_prior = distributions.Dirichlet(**gating_hypparams)

        dpglms, scores, test_evars = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                                              train_data=train_data,
                                                              test_data=test_data,
                                                              input_dim=input_dim,
                                                              output_dim=output_dim,
                                                              arguments=args)

        # write raw data to file
        _used_labels = [len(dpglm.used_labels) for dpglm in dpglms]
        with open(raw_path, 'w+') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(test_evars, _used_labels))

        # write stats data to file
        _test_evars = np.asarray(test_evars)
        _test_evars_mean = np.mean(_test_evars, axis=0)
        _test_evars_std = np.std(_test_evars, axis=0)
        _test_evars_var = np.var(_test_evars, axis=0)
        _test_evars_median = np.median(_test_evars, axis=0)
        _test_evars_q1 = np.quantile(_test_evars, 0.25, axis=0)
        _test_evars_q3 = np.quantile(_test_evars, 0.75, axis=0)

        file = open(stats_path, 'w+')
        file.write('mean' + ' ' + str(_test_evars_mean) + '\n')
        file.write('std' + ' ' + str(_test_evars_std) + '\n')
        file.write('var' + ' ' + str(_test_evars_var) + '\n')
        file.write('median' + ' ' + str(_test_evars_median) + '\n')
        file.write('q1' + ' ' + str(_test_evars_q1) + '\n')
        file.write('q3' + ' ' + str(_test_evars_q3) + '\n')
        file.close()

        if itr == 0:
            violin_data = _test_evars
        else:
            violin_data = np.column_stack((violin_data, _test_evars))

    # total script runtime
    stop = timeit.default_timer()
    overall_time = stop - start

    # set paths for tikz and pdf
    tikz_path = os.path.join(str(args.dataset) + '/tikz/' + str(args.dataset) + '_' + str(tmp) + time)
    pdf_path = os.path.join(str(args.dataset) + '/pdf/' + str(args.dataset) + '_' + str(tmp) + time)

    # generate and save violin plots
    nb_cols = len(alphas)

    x_label = None
    if args.prior == 'dirichlet':
        x_label = 'Alpha of Dirichlet Prior'
    elif args.prior == 'stick-breaking':
        x_label = 'Alpha of Stick-breaking Prior'

    x_categories = [str(alpha) for alpha in alphas]
    y_label = 'Explained Variance Score'
    title = None

    plot_violin_box(violin_data, nb_cols, tikz_path, pdf_path, x_label, y_label, title, x_categories)
