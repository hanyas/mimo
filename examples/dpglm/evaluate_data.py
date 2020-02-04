import numpy as np
import numpy.random as npr
import scipy.stats as stats

import mimo
from mimo import distributions, models
from mimo.util.data import load_data, trajectory_data, noise_function
from mimo.util.prediction import sample_prediction, single_prediction, single_trajectory_prediction
from mimo.util.prediction import em_prediction, meanfield_prediction, gibbs_prediction, gibbs_prediction_noWeights, \
    meanfield_traj_prediction
from mimo.util.plot import plot_violin_box

import os
import timeit
import datetime
import argparse
import csv

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from sklearn.metrics import explained_variance_score, mean_squared_error

from joblib import Parallel, delayed
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

    # initialize Normal-Inverse-Wishart of input
    mu_low = np.amin(train_data[:, :-output_dim])
    mu_high = np.amax(train_data[:, :-output_dim])

    psi_niw = npr.uniform(0, 10)
    kappa = npr.uniform(0, 0.1)

    # initialize Matrix-Normal-Inverse-Wishart of output
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

    # Gibbs sampling to wander around the posterior
    for _ in range(args.gibbs_iters):
        dpglm.resample_model()

    # Mean field
    score.append(dpglm.meanfield_coordinate_descent(tol=args.earlystop,
                                                    maxiter=args.meanfield_iters,
                                                    progprint=False))

    # set plot style to seaborn
    sns.set()

    # marginal prediction
    if args.trajectory:

        # marginal prediction
        mean, var, test_mse, test_evar = meanfield_traj_prediction(dpglm, test_data, input_dim, output_dim,
                                                                   args.traj_step, args.mode_prediction,
                                                                   args.traj_trick, prior=args.prior)
        # plot single prediction on training data
        single_trajectory_prediction(dpglm, train_data, train_data_saved, args.traj_trick, output_dim, input_dim)

        # plot mean / mode prediction on test data
        if output_dim == 1:
            plt.scatter(test_data_saved[:, 0], test_data[:, 0], s=1) # plot true position
            plt.plot(test_data_saved[:, 0], mean[:, 0], color='red') # plot estimated position
        else:
            plt.scatter(test_data_saved[:, 0], test_data[:, 0], s=1)  # plot true position
            plt.scatter(test_data_saved[:, 0], test_data[:, 1], s=1) # plot true velocity
            plt.plot(test_data_saved[:, 0], mean, color='red') # plot estimated velocity and position
        plt.show()

        # # plot test_evar over prediction horizon
        # plt.plot(np.arange(len(test_evar))+1, test_evar, '-o')
        # plt.title('Explained variance score over prediction horizon')
        # plt.xlabel('prediction horizon')
        # plt.ylabel('explained variance score')
        # plt.show()
        #
        # # plot test_mse over prediction horizon
        # plt.show()
        # plt.plot(np.arange(len(test_mse))+1, test_mse, '-o')
        # plt.title('MSE over prediction horizon')
        # plt.xlabel('prediction horizon')
        # plt.ylabel('MSE')
        # plt.show()

    else:

        # marginal prediction
        mean, var, noise_std = meanfield_prediction(dpglm, test_data, input_dim, output_dim, args.mode_prediction, prior=args.prior)
        # mean, var = em_prediction(dpglm, test_data, input_dim, output_dim)
        # mean, var = gibbs_prediction(dpglm, test_data, train_data, input_dim, output_dim, args.gibbs_samples, args.prior, args.affine)
        # mean, var = gibbs_prediction_noWeights(dpglm, test_data, train_data, input_dim, output_dim, args.gibbs_samples, args.prior, args.affine)

        # plot single prediction on training data
        single_prediction(dpglm, train_data, input_dim, output_dim)  # show prediction for a single sample from posterior

        # plot mean or mode prediction on test data
        sorting = np.argsort(test_data, axis=0)  # sort based on input values
        sorted_data = np.take_along_axis(test_data, sorting, axis=0)
        sorted_mean = np.take_along_axis(mean, sorting[:, [0]], axis=0)
        sorted_var = np.take_along_axis(var, sorting[:, [0]], axis=0)
        plt.scatter(test_data[:, 0], test_data[:, 1], s=1)
        plt.plot(sorted_data[:, 0], sorted_mean[:, 0], color='red')
        plt.plot(sorted_data[:, 0], sorted_mean[:, 0] + 2. * np.sqrt(sorted_var[:, 0]), color='green')
        plt.plot(sorted_data[:, 0], sorted_mean[:, 0] - 2. * np.sqrt(sorted_var[:, 0]), color='green')
        plt.show()

        # # create subplots
        # fig = plt.figure()
        # gs = gridspec.GridSpec(3, 1, height_ratios=[6, 1, 1])
        # xrange = [np.amin(test_data[:, 0]), np.amax(test_data[:, 0])] # set range for plot of noise level and activations
        #
        # # plot data generation noise level and estimated noise level
        # ax0 = plt.subplot(gs[0])
        # true_xvals, true_noise = noise_function(n_train, 0.1, 0.3, xrange)
        # ax0.plot(true_xvals, true_noise, color='red')
        # sorting = np.argsort(test_data, axis=0)  # sort based on input values
        # sorted_data = np.take_along_axis(test_data, sorting, axis=0)
        # sorting = np.atleast_2d(sorting)
        # sorted_noise_std = np.take_along_axis(noise_std, sorting[:, [0]], axis=0)
        # ax0.plot(sorted_data[:, 0], sorted_noise_std, color='green')
        #
        # # plot mean or mode prediction on test data
        # ax1 = plt.subplot(gs[1])
        # sorting = np.argsort(test_data, axis=0)  # sort based on input values
        # sorted_data = np.take_along_axis(test_data, sorting, axis=0)
        # sorted_mean = np.take_along_axis(mean, sorting[:, [0]], axis=0)
        # sorted_var = np.take_along_axis(var, sorting[:, [0]], axis=0)
        # ax1.scatter(test_data[:, 0], test_data[:, 1], s=1)
        # ax1.plot(sorted_data[:, 0], sorted_mean, color='red')
        # ax1.plot(sorted_data[:, 0], sorted_mean + 2. * np.sqrt(sorted_var), color='green')
        # ax1.plot(sorted_data[:, 0], sorted_mean - 2. * np.sqrt(sorted_var), color='green')
        #
        # # plot gaussian activations
        # ax2 = plt.subplot(gs[2])
        # x_mu, x_sigma = [], []
        # for idx, c in enumerate(dpglm.components):
        #     if idx in dpglm.used_labels:
        #         mu, kappa, psi_niw, nu_niw, Mk, Vk, psi_mniw, nu_mniw = c.posterior.params
        #         sigma = np.sqrt(1 / kappa * psi_niw)
        #         x_mu.append(mu[0])
        #         x_sigma.append(sigma[0])
        # # x_mu, x_sigma = np.asarray(x_mu), np.asarray(x_sigma)
        # for i in range(len(dpglm.used_labels)):
        #     x = np.linspace(xrange[0], xrange[1], 100)
        #     ax2.plot(x, stats.norm.pdf(x, x_mu[i], x_sigma[i]))
        #
        # # plt.tight_layout()
        # plt.show()

    if not args.trajectory:
        test_mse = mean_squared_error(test_data[:, input_dim:], mean)
        test_evar = explained_variance_score(test_data[:, input_dim:], mean, multioutput='variance_weighted')

    print('test_evar', test_evar)
    print('test_mse', test_mse)

    return dpglm, score, test_evar


def parallel_dpglm_inference(nb_jobs=50, **kwargs):
    kwargs_list = [kwargs for _ in range(nb_jobs)]
    results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
    dpglms, scores, test_evars = list(map(list, zip(*results)))
    return dpglms, scores, test_evars


if __name__ == "__main__":

    # timer
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Evaluate DPGLM with a Stick-breaking prior')
    parser.add_argument('--dataset', help='Choose dataset', default='ball')
    parser.add_argument('--traintest_ratio', help='Set ratio of training to test data', default=5, type=float)
    parser.add_argument('--datapath', help='Set path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='Set path to dataset', default=os.path.abspath(mimo.__file__ + '/../../evaluation'))
    parser.add_argument('--nb_seeds', help='Set number of seeds', default=1, type=int)
    parser.add_argument('--prior', help='Set prior type', default='stick-breaking')
    parser.add_argument('--alpha', help='Set concentration parameter', default=100, type=float)
    parser.add_argument('--nb_models', help='Set max number of models', default=10, type=int)
    parser.add_argument('--affine', help='Set affine or not', default=True, type=bool)
    parser.add_argument('--gibbs_iters', help='Set Gibbs iterations', default=250, type=int)
    parser.add_argument('--gibbs_samples', help='Set number of Gibbs samples', default=5, type=int)
    parser.add_argument('--meanfield_iters', help='Set max. VI iterations', default=500, type=int)
    parser.add_argument('--mode_prediction', help='Set VI prediction to mode or not (=mean)', default=True, type=bool)
    parser.add_argument('--earlystop', help='Set stopping criterion for VI', default=1e-2, type=float)
    parser.add_argument('--trajectory', help='Set dataset and prediction to trajectory', default=True, type=bool)
    parser.add_argument('--traj_step', help='Set step for trajectory prediction', default=1, type=int)
    parser.add_argument('--traj_trick', help='Force trajectory prediction to stay close to previous input', default=True, type=bool)

    args = parser.parse_args()

    # prepare dataset
    nb_samples = []
    data_file = args.dataset + '.csv'  # name of the file within datasets/
    if args.dataset == 'cmb':
        # nb_samples = [100, 200, 300, 400, 600]
        nb_samples = [600]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'sine':
        # nb_samples = [250, 500, 750, 1000, 1500]
        nb_samples = [1500]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'fk_1joint':
        nb_samples = [30, 100, 250, 500, 1000]
        input_dim = 1
        output_dim = 2
    elif args.dataset == 'fk_2joint':
        nb_samples = [30, 100, 250, 500, 1000]
        input_dim = 2
        output_dim = 2
    elif args.dataset == 'fk_3joint':
        nb_samples = [500, 1000, 1500, 2000, 2500]
        input_dim = 3
        output_dim = 2
    elif args.dataset == 'sarcos':
        nb_samples = [500, 1000, 1500, 2000, 2500]
        input_dim = 21
        output_dim = 7
    elif args.dataset == 'step_noise':
        nb_samples = [300]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'step':
        nb_samples = [300]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'step_polynomial_deg3':
        nb_samples = [1000]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'step_polynomial_deg3_v2':
        nb_samples = [1000]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'step_polynomial_deg3_v3':
        nb_samples = [1000]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'step_polynomial_deg3_v4':
        nb_samples = [1000]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'step_polynomial_deg3_v5':
        nb_samples = [1000]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'step_polynomial_deg3_v6':
        nb_samples = [1000]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'sine_noise_step_a0':
        nb_samples = [500]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'sine_noise_step_a1':
        nb_samples = [500]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'sine_noise_step_b1':
        nb_samples = [500]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'sine_noise_expsin':
        nb_samples = [2000]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'sine_noise_linear':
        nb_samples = [2000]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'sine_noise_sigmoids':
        nb_samples = [2000]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'ball':
        nb_samples = [2000]  # 6900 total samples
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'ball_vel':
        nb_samples = [2000]
        input_dim = 1
        output_dim = 2
    elif args.dataset == 'ball_vel_v2':
        nb_samples = [2000]
        input_dim = 1
        output_dim = 2
    elif args.dataset == 'ball_vel_v3':
        nb_samples = [2000]
        input_dim = 1
        output_dim = 2
    elif args.dataset == 'ball_vel_v4':
        nb_samples = [900]
        input_dim = 1
        output_dim = 2
    elif args.dataset == 'silverman':
        nb_samples = [94] # 94 total samples
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'goldberg':
        nb_samples = [100]
        input_dim = 1
        output_dim = 1
    elif args.dataset == 'yuan':
        nb_samples = [2000]
        input_dim = 1
        output_dim = 1
    else:
        raise RuntimeError("Dataset does not exist")

    # save time stamp for file names
    time = str(datetime.datetime.now().strftime('_%m-%d_%H-%M-%S'))

    # evaluation criteria
    eval_str = 'datasize'

    violin_data_scores = None
    violin_data_labels = None

    for itr, n_train in enumerate(nb_samples):
        print('Current size of dataset', n_train)

        # load dataset
        n_test = int(n_train / args.traintest_ratio)
        train_data, test_data = load_data(n_train, n_test,
                                          data_file, args.datapath,
                                          output_dim, input_dim,
                                          args.dataset == 'sarcos',
                                          seed=1337)

        # convert to trajectory dataset
        train_data_saved = np.copy(train_data)
        test_data_saved = np.copy(test_data)
        if args.trajectory:
            train_data = trajectory_data(train_data, output_dim, input_dim, args.traj_trick)
            test_data = trajectory_data(test_data, output_dim, input_dim, args.traj_trick)
            input_dim = output_dim

        # set working directory
        os.chdir(args.evalpath)

        raw_path = os.path.join(str(args.dataset) + '/raw/' + str(args.dataset) + '_' + str(eval_str) + '_' + str(n_train) + '_raw' + time + '.csv')
        scores_stats_path = os.path.join(str(args.dataset) + '/stats/' + str(args.dataset) + '_' + str(eval_str) + '_' + str(n_train) + '_scores_stats' + time + '.csv')
        labels_stats_path = os.path.join(str(args.dataset) + '/stats/' + str(args.dataset) + '_' + str(eval_str) + '_' + str(n_train) + '_labels_stats' + time + '.csv')

        # define gating
        if args.prior == 'stick-breaking':
            gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models, )), deltas=np.ones((args.nb_models, )) * args.alpha)
            gating_prior = distributions.StickBreaking(**gating_hypparams)
        else:
            gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models, )) * args.alpha)
            gating_prior = distributions.Dirichlet(**gating_hypparams)

        dpglms, scores, test_evars = parallel_dpglm_inference(nb_jobs=args.nb_seeds,
                                                              train_data=train_data,
                                                              test_data=test_data,
                                                              input_dim=input_dim,
                                                              output_dim=output_dim,
                                                              arguments=args)

        # # write raw data to file
        used_labels = [len(dpglm.used_labels) for dpglm in dpglms]
        print('used_labels',used_labels)
        # with open(raw_path, 'w+') as f:
        #     writer = csv.writer(f, delimiter='\t')
        #     writer.writerows(zip(test_evars, used_labels))
        #
        # # write explained variance stats data to file
        # _test_evars = np.asarray(test_evars)
        # _test_evars_mean = np.mean(_test_evars, axis=0)
        # _test_evars_std = np.std(_test_evars, axis=0)
        # _test_evars_var = np.var(_test_evars, axis=0)
        # _test_evars_median = np.median(_test_evars, axis=0)
        # _test_evars_q1 = np.quantile(_test_evars, 0.25, axis=0)
        # _test_evars_q3 = np.quantile(_test_evars, 0.75, axis=0)
        #
        # file = open(scores_stats_path, 'w+')
        # file.write('mean' + ' ' + str(_test_evars_mean) + '\n')
        # file.write('std' + ' ' + str(_test_evars_std) + '\n')
        # file.write('var' + ' ' + str(_test_evars_var) + '\n')
        # file.write('median' + ' ' + str(_test_evars_median) + '\n')
        # file.write('q1' + ' ' + str(_test_evars_q1) + '\n')
        # file.write('q3' + ' ' + str(_test_evars_q3) + '\n')
        # file.close()
        #
        # if itr == 0:
        #     violin_data_scores = _test_evars
        # else:
        #     violin_data_scores = np.column_stack((violin_data_scores, _test_evars))

        # write label stats data to file
        _used_labels = np.asarray(used_labels)
        _used_labels_mean = np.mean(_used_labels, axis=0)
        _used_labels_std = np.std(_used_labels, axis=0)
        _used_labels_var = np.var(_used_labels, axis=0)
        _used_labels_median = np.median(_used_labels, axis=0)
        _used_labels_q1 = np.quantile(_used_labels, 0.25, axis=0)
        _used_labels_q3 = np.quantile(_used_labels, 0.75, axis=0)

        # file = open(labels_stats_path, 'w+')
        # file.write('mean' + ' ' + str(_used_labels_mean) + '\n')
        # file.write('std' + ' ' + str(_used_labels_std) + '\n')
        # file.write('var' + ' ' + str(_used_labels_var) + '\n')
        # file.write('median' + ' ' + str(_used_labels_median) + '\n')
        # file.write('q1' + ' ' + str(_used_labels_q1) + '\n')
        # file.write('q3' + ' ' + str(_used_labels_q3) + '\n')
        # file.close()

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
    nb_cols = len(nb_samples)

    x_label = 'Training Sample Size'

    x_categories = [str(n_train) for n_train in nb_samples]
    scores_y_label = 'Explained Variance Score'
    labels_y_label = 'Number of Linear Models'
    title = None

    # plot_violin_box(violin_data_scores, nb_cols, scores_tikz_path, scores_pdf_path, x_label, scores_y_label, title, x_categories)
    # plot_violin_box(violin_data_labels, nb_cols, labels_tikz_path, labels_pdf_path, x_label, labels_y_label, title, x_categories)
