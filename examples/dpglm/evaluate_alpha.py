import numpy as np
import numpy.random as npr

import mimo
from mimo import distributions, models
from mimo.util.data import load_data
from mimo.util.prediction import sample_prediction, single_prediction
from mimo.util.prediction import em_prediction, meanfield_prediction
from mimo.util.plot import plot_violin_box

import os
import timeit
import datetime
import argparse

from joblib import Parallel, delayed

from sklearn.metrics import explained_variance_score
import csv

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
    mean, var, score = None, None,  []

    if args.inference == 'em':
        # Expectation Maximizaiton
        for _ in range(args.em_iters):
            dpglm.EM_step()

        # EM testing prediction
        mean, var = em_prediction(dpglm, test_data, input_dim, output_dim)

    elif args.inference == 'gibbs':
        # Gibbs sampling
        for _ in range(args.gibbs_iters):
            dpglm.resample_model()

    elif args.inference == 'meanfield':
        # Mean field
        score.append(dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                                        maxiter=args.meanfield_iters,
                                                        progprint=False))
        # meanfield marginal prediction
        mean, var = meanfield_prediction(dpglm, test_data, input_dim, output_dim, prior=args.prior)

    elif args.inference == 'gibbs-meanfield':
        # Gibbs sampling
        for _ in range(args.gibbs_iters):
            dpglm.resample_model()

        # Mean field
        score.append(dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                                        maxiter=args.meanfield_iters,
                                                        progprint=False))
        # marginal prediction
        mean, var = meanfield_prediction(dpglm, test_data, input_dim, output_dim, prior=args.prior)

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
    parser.add_argument('--dataset', help='Choose dataset', default='sine')
    parser.add_argument('--datapath', help='Set path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='Set path to dataset', default=os.path.abspath(mimo.__file__ + '/../../evaluation'))
    parser.add_argument('--nb_seeds', help='Set number of seeds', default=100, type=int)
    parser.add_argument('--prior', help='Set prior type', default='dirichlet')
    parser.add_argument('--nb_models', help='Set max number of models', default=100, type=int)
    parser.add_argument('--affine', help='Set affine or not', default=True, type=bool)
    parser.add_argument('--inference', help='Set inference technique', default='gibbs-meanfield')
    parser.add_argument('--em_iters', help='Set EM iterations', default=1000, type=int)
    parser.add_argument('--gibbs_iters', help='Set Gibbs iterations', default=1000, type=int)
    parser.add_argument('--meanfield_iters', help='Set VI iterations', default=500, type=int)
    parser.add_argument('--earlystop', help='Set stopping criterion for VI', default=1e-2, type=float)

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
    eval_str = 'alpha_' + str(args.prior)

    violin_data_scores = None
    violin_data_labels = None

    for itr, alpha in enumerate(alphas):
        print('Current alpha value', alpha)

        raw_path = os.path.join(str(args.dataset) + '/raw/' + str(args.dataset) + '_' + str(eval_str) + '_' + str(alpha) + '_raw' + time + '.csv')
        scores_stats_path = os.path.join(str(args.dataset) + '/stats/' + str(args.dataset) + '_' + str(eval_str) + '_' + str(alpha) + '_scores_stats' + time + '.csv')
        labels_stats_path = os.path.join(str(args.dataset) + '/stats/' + str(args.dataset) + '_' + str(eval_str) + '_' + str(alpha) + '_labels_stats' + time + '.csv')

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
        used_labels = [len(dpglm.used_labels) for dpglm in dpglms]
        with open(raw_path, 'w+') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(test_evars, used_labels))

        # write stats data to file
        _test_evars = np.asarray(test_evars)
        _test_evars_mean = np.mean(_test_evars, axis=0)
        _test_evars_std = np.std(_test_evars, axis=0)
        _test_evars_var = np.var(_test_evars, axis=0)
        _test_evars_median = np.median(_test_evars, axis=0)
        _test_evars_q1 = np.quantile(_test_evars, 0.25, axis=0)
        _test_evars_q3 = np.quantile(_test_evars, 0.75, axis=0)

        file = open(scores_stats_path, 'w+')
        file.write('mean' + ' ' + str(_test_evars_mean) + '\n')
        file.write('std' + ' ' + str(_test_evars_std) + '\n')
        file.write('var' + ' ' + str(_test_evars_var) + '\n')
        file.write('median' + ' ' + str(_test_evars_median) + '\n')
        file.write('q1' + ' ' + str(_test_evars_q1) + '\n')
        file.write('q3' + ' ' + str(_test_evars_q3) + '\n')
        file.close()

        if itr == 0:
            violin_data_scores = _test_evars
        else:
            violin_data_scores = np.column_stack((violin_data_scores, _test_evars))

        # write label stats data to file
        _used_labels = np.asarray(used_labels)
        _used_labels_mean = np.mean(_used_labels, axis=0)
        _used_labels_std = np.std(_used_labels, axis=0)
        _used_labels_var = np.var(_used_labels, axis=0)
        _used_labels_median = np.median(_used_labels, axis=0)
        _used_labels_q1 = np.quantile(_used_labels, 0.25, axis=0)
        _used_labels_q3 = np.quantile(_used_labels, 0.75, axis=0)

        file = open(labels_stats_path, 'w+')
        file.write('mean' + ' ' + str(_used_labels_mean) + '\n')
        file.write('std' + ' ' + str(_used_labels_std) + '\n')
        file.write('var' + ' ' + str(_used_labels_var) + '\n')
        file.write('median' + ' ' + str(_used_labels_median) + '\n')
        file.write('q1' + ' ' + str(_used_labels_q1) + '\n')
        file.write('q3' + ' ' + str(_used_labels_q3) + '\n')
        file.close()

        if itr == 0:
            violin_data_labels = _used_labels
        else:
            violin_data_labels = np.column_stack((violin_data_labels, _used_labels))

    # total script runtime
    stop = timeit.default_timer()
    overall_time = stop - start

    # set paths for tikz and pdf
    scores_tikz_path = os.path.join(str(args.dataset) + '/tikz/' + str(args.dataset) + '_' + str(eval_str) + '_scores' + time)
    scores_pdf_path = os.path.join(str(args.dataset) + '/pdf/' + str(args.dataset) + '_' + str(eval_str) + '_scores' + time)
    labels_tikz_path = os.path.join(str(args.dataset) + '/tikz/' + str(args.dataset) + '_' + str(eval_str) + '_labels' + time)
    labels_pdf_path = os.path.join(str(args.dataset) + '/pdf/' + str(args.dataset) + '_' + str(eval_str) + '_labels' + time)

    # generate and save violin plots
    nb_cols = len(alphas)

    x_label = None
    if args.prior == 'dirichlet':
        x_label = 'Alpha of Dirichlet Prior'
    elif args.prior == 'stick-breaking':
        x_label = 'Alpha of Stick-breaking Prior'

    x_categories = [str(alpha) for alpha in alphas]
    scores_y_label = 'Explained Variance Score'
    labels_y_label = 'Number of Linear Models'
    title = None

    plot_violin_box(violin_data_scores, nb_cols, scores_tikz_path, scores_pdf_path, x_label, scores_y_label, title, x_categories)
    plot_violin_box(violin_data_labels, nb_cols, labels_tikz_path, labels_pdf_path, x_label, labels_y_label, title, x_categories)
