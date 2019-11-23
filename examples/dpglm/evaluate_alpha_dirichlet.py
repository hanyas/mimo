import numpy as np
import numpy.random as npr

from sklearn.metrics import explained_variance_score
import pandas

import mimo
from mimo import distributions, models
from mimo.util.text import progprint_xrange
from mimo.util.data import *
from mimo.util.plot import *
from mimo.util.prediction import *

import os
import operator
import copy
import timeit
import datetime
import argparse

# set random seed
np.random.seed(seed=None)

# timer
start = timeit.default_timer()

parser = argparse.ArgumentParser(description='Evaluate DPGLM with a Dirichlet prior')
parser.add_argument('--dataset', help='Choose dataset', default='cmb')
parser.add_argument('--datapath', help='Set path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
parser.add_argument('--prior', help='Set prior type', default='dirichlet')
parser.add_argument('--inference', help='Set inference technique', default='gibbs')
parser.add_argument('--iterations', help='Set inference iterations', default=300)
parser.add_argument('--nb_seeds', help='Set number of seeds', default=100)
parser.add_argument('--nb_models', help='Set max number of models', default=100)
parser.add_argument('--affine', help='Set affine or not', default=True)

args = parser.parse_args()

data_file = args.dataset + '.csv'  # name of the file within datasets/
if args.dataset == 'cmb':
    n_train = 600
    in_dim_niw = 1
    out_dim = 1
elif args.dataset == 'sine':
    n_train = 1500
    in_dim_niw = 1
    out_dim = 1
elif args.dataset == 'fk_1joint':
    n_train = 1000
    in_dim_niw = 1
    out_dim = 2
elif args.dataset == 'fk_2joint':
    n_train = 1000
    in_dim_niw = 2
    out_dim = 2
elif args.dataset == 'fk_3joint':
    n_train = 2500
    in_dim_niw = 3
    out_dim = 2
elif args.dataset == 'sarcos':
    n_train = 2500
    in_dim_niw = 21
    out_dim = 7
else:
    raise RuntimeError("Dataset does not exist")

# concentration parameter of the prior
alphas = [0.01, 0.1, 1., 5., 10., 50., 100.]

# number of seeds
metaitr = args.nb_seeds

# generate and save violin plots
num_cols = len(alphas)
x_label = 'Alpha of Dirichlet Prior'
x_categories = ['0.01', '0.1', '1', '5', '10', '50', '100']
y_label = 'Explained Variance Score'
title = None

for alpha in alphas:
    print('Current alpha value', alpha)

    n_test = int(n_train / 5)
    data, data_test = load_data(n_train, n_test, data_file, args.datapath, out_dim, in_dim_niw, args.dataset=='sarcos')

    superitr = 1

    # set working directory and file name
    os.chdir(args.datapath)
    str1 = 'alpha_' + str(args.prior)
    str2 = alpha
    time = str(datetime.datetime.now().strftime('_%m-%d_%H-%M-%S'))

    csv_path = os.path.join('evaluation/' + str(args.dataset) + '/raw/' + str(args.dataset) + '_' + str(str1) + '_' + str(str2) + '_raw' + time + '.csv')
    tikz_path = os.path.join('evaluation/' + str(args.dataset) + '/tikz/' + str(args.dataset) + '_' + str(str1) + '_' + str(str2) + time + '.tex')
    pdf_path = os.path.join('evaluation/' + str(args.dataset) + '/pdf/' + str(args.dataset) + '_' + str(str1) + '_' + str(str2) + time + '.pdf')
    stat_path = os.path.join('evaluation/' + str(args.dataset) + '/stats/' + str(args.dataset) + '_' + str(str1) + '_' + str(str2) + '_stats' + time + '.csv')
    visual_tikz_path = os.path.join('evaluation/' + str(args.dataset) + '/visual/' + str(args.dataset) + '_' + str(str1))
    visual_pdf_path = os.path.join('evaluation/' + str(args.dataset) + '/visual/' + str(args.dataset) + '_' + str(str1))

    # inference settings
    gibbs = True
    gibbs_iter = 300

    mf_conv = True  # mf with convergence criterion and max_iter

    mf = False      # mf with fixed iterations
    mf_iter = 150

    mf_sgd, batch_size, epochs, step_size = False, 30, 300, 1e-1

    # define gating
    if args.prior == 'stick-breaking':
        gating_hypparams = dict(K=args.nb_models, gammas=np.ones((args.nb_models, )), deltas=np.ones((args.nb_models, )) * alpha)
        gating_prior = distributions.StickBreaking(**gating_hypparams)
    else:
        gating_hypparams = dict(K=args.nb_models, alphas=np.ones((args.nb_models, )) * alpha)
        gating_prior = distributions.Dirichlet(**gating_hypparams)

    for m in range(args.nb_seeds):
        print('meta iter', m)

        # initialize prior parameters: draw from uniform disitributions
        components_prior = []
        for j in range(args.nb_models):
            in_dim_mniw = in_dim_niw
            if args.affine:
                in_dim_mniw = in_dim_niw + 1

            V = np.eye(in_dim_mniw)
            for i in range(in_dim_mniw):
                if i < in_dim_mniw - 1:
                    V[i, i] = npr.uniform(0, 10)
                else:
                    V[i, i] = npr.uniform(0, 1000)  # offset
            rnd_psi_mniw = npr.uniform(0, 0.1)
            nu_mniw = 2 * out_dim + 1

            mu_low = np.amin(data[:, :-out_dim])
            mu_high = np.amax(data[:, :-out_dim])

            rnd_psi_niw = npr.uniform(0, 10)
            rnd_kappa = npr.uniform(0, 0.1)

            components_hypparams = dict(mu=npr.uniform(mu_low, mu_high, size=in_dim_niw),
                                        kappa=rnd_kappa,  # 0.05
                                        psi_niw=np.eye(in_dim_niw) * rnd_psi_niw,
                                        nu_niw=2 * in_dim_niw + 1,
                                        M=np.zeros((out_dim, in_dim_mniw)), V=V,
                                        affine=args.affine,
                                        psi_mniw=np.eye(out_dim) * rnd_psi_mniw,
                                        nu_mniw=nu_mniw)
            components_prior_rand = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)
            components_prior.append(components_prior_rand)

        # define model
        if args.prior == 'stick-breaking':
            dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                                   components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
        else:
            dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                                   components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(args.nb_models)])
        dpglm.add_data(data)

        start_inference = timeit.default_timer()

        # inference
        all_scores = []
        all_models = []
        all_nmse = []
        all_err = []
        for _ in range(superitr):
            scores = []
            nmse = []
            err = []

            # Gibbs sampling to wander around the posterior
            if gibbs:
                # print('Gibbs Sampling')
                # for _ in progprint_xrange(gibbs_iter):
                for _ in range(gibbs_iter):
                    dpglm.resample_model()
                    # this is needed if Gibbs sampling is used without Mean Field
                    if not mf and not mf_conv and not mf_sgd:
                        for idx, l in enumerate(dpglm.labels_list):
                            l.r = l.get_responsibility()
                        scores.append(dpglm._vlb())
            if mf:
                # mean field to lock onto a mode (without convergence criterion, with fixed number of iterations)
                # print('Mean Field')
                if not gibbs:
                    dpglm.resample_model()
                # scores = [dpglm.meanfield_coordinate_descent_step() for _ in progprint_xrange(250)]
                # for _ in progprint_xrange(mf_iter):
                for _ in range(mf_iter):
                    scores.append(dpglm.meanfield_coordinate_descent_step())

            # pred_y, err, err_squared = predict_train(dpglm, data, out_dim, in_dim_niw)
            # var = np.var(data[:, in_dim_niw:], axis=0)
            # MSE = 1 / n_train * err_squared
            # nmse.append(np.sum(MSE[0] / var[0]))
            # err_.append(np.sum(err))

            if mf_conv:
                # mean field to lock onto a mode (with convergence criterion)
                # print('Mean Field')
                if not gibbs:
                    dpglm.resample_model()
                scores = dpglm.meanfield_coordinate_descent(tol=1e-2, maxiter=500, progprint=False)

            # stochastic mean field to lock onto a mode
            if mf_sgd:
                # print('Stochastic Mean Field')
                if not gibbs:
                    dpglm.resample_model()
                minibatchsize = batch_size
                prob = minibatchsize / float(n_train)
                # for _ in progprint_xrange(epochs):
                for _ in range(epochs):
                    minibatch = npr.permutation(n_train)[:minibatchsize]
                    dpglm.meanfield_sgdstep(minibatch=data[minibatch], prob=prob, stepsize=step_size)
                    for idx, l in enumerate(dpglm.labels_list):
                        l.r = l.get_responsibility()
                    scores.append(dpglm._vlb())
                # all_scores.append(dpglm.meanfield_coordinate_descent_step())

            # all_err.append(err_)
            # all_nmse.append(nmse)
            all_scores.append(scores)
            all_models.append(copy.deepcopy(dpglm))

        start_plotting = timeit.default_timer()

        # Sort models and select best one
        models_and_scores = sorted([(m, s[-1]) for m, s in zip(all_models, all_scores)],
                                   key=operator.itemgetter(1), reverse=True)
        dpglm = models_and_scores[0][0]
        vlb = models_and_scores[0][1]

        # get labels
        for l in dpglm.labels_list:
            label = l.z
        # print('used labels',dpglm.used_labels)
        # print('# used labels',len(dpglm.used_labels))

        # predict on training data
        pred_y, err, err_squared = predict_train(dpglm, data, out_dim, in_dim_niw)

        # calculate error metrics on training data
        explained_var_train = explained_variance_score(data[:, in_dim_niw:], pred_y, multioutput='variance_weighted')

        start_prediction = timeit.default_timer()

        # calculate alpha_hat (sum over alpha_k)
        if not stick_breaking:
            alphas_hat = 0
            alphas = dpglm.gating.posterior.alphas
            for idx, c in enumerate(dpglm.components):
                if idx in dpglm.used_labels:
                    alphas_hat = alphas_hat + alphas[idx]
        else:
            stick_lengths = np.ones([len(dpglm.components)])
            product = np.ones([len(dpglm.components)])
            gammas = dpglm.gating.posterior.gammas
            deltas = dpglm.gating.posterior.deltas
            for idx, c in enumerate(dpglm.components):
                if idx in dpglm.used_labels:
                    product[idx] = gammas[idx] / gammas[idx] + deltas[idx]
                    for j in range(idx):
                        product[idx] = product[idx] * gammas[j] / gammas[j] + deltas[j]
            alphas = product

        # initialize variables
        mean_function, plus_std_function, minus_std_function = np.empty_like(data_test[:, in_dim_niw:]),\
                                                               np.empty_like(data_test[:, in_dim_niw:]),\
                                                               np.empty_like(data_test[:, in_dim_niw:])

        marg, stud_t = np.zeros([len(dpglm.components)]), np.zeros([len(dpglm.components)])
        err, err_squared = 0, 0
        dot_xx = np.zeros([len(dpglm.components), in_dim_mniw, in_dim_mniw])
        dot_yx = np.zeros([len(dpglm.components), out_dim, in_dim_mniw])
        dot_yy = np.zeros([len(dpglm.components), out_dim, out_dim])
        n = np.zeros([len(dpglm.components)])

        # get statistics for each component for training data
        for idx, c in enumerate(dpglm.components):
            statistics = c.posterior.get_statistics([l.data[l.z == idx] for l in dpglm.labels_list])
            dot_yx[idx], dot_xx[idx], dot_yy[idx], n[idx] = statistics[4], statistics[5], statistics[6], statistics[7]

        # prediction / mean function of y_hat for all training data x_hat
        for i in range(len(data_test[:, :-out_dim])):
            x_hat = data_test[i, :-out_dim]
            y_hat = data_test[i, in_dim_niw:]

            # calculate the marginal likelihood of training data x_hat for each cluster (under NIW distribution)
            # calculate the normalization term for mean function (= alphas_marg_sum) for x_hat
            alphas_marg_sum = 0
            for idx, c in enumerate(dpglm.components):
                if idx in dpglm.used_labels:
                    mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw = get_component_standard_parameters(c.posterior)
                    marg[idx] = NIW_marg_likelihood(x_hat, mu, kappa, psi_niw, nu_niw, 1, out_dim)
                    alphas_marg_sum = alphas_marg_sum + alphas[idx] * marg[idx]
                    # stud_t[idx] = student_t(x_hat, mu, kappa, psi_niw, nu_niw, in_dim_niw)

            # calculate contribution of each cluster to mean function / prediction for training data x_hat
            term_mean = 0
            term_plus_std = 0
            term_minus_std = 0
            for idx, c in enumerate(dpglm.components):
                if idx in dpglm.used_labels:
                    mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw = get_component_standard_parameters(c.posterior)
                    S_0, N_0 = get_component_standard_parameters(c.prior)[6], get_component_standard_parameters(c.prior)[7]
                    mat_T, std_T = matrix_t(data, idx, label, out_dim, in_dim_niw, affine,
                                            x_hat, V, M, args.nb_models, S_0, dot_xx[idx],
                                            dot_yx[idx], dot_yy[idx], psi_mniw, nu_mniw, N_0)

                    # mean function
                    term_mean = term_mean + mat_T * marg[idx] * alphas[idx] / alphas_marg_sum
                    # term_mean = term_mean + mat_T * marg[idx] / marg.sum()
                    # term = term + mat_T * alphas[idx] * marg[idx]  / alphas_sum * len(dpglm.components)

                    # confidence interval
                    term_plus_std = term_plus_std + (mat_T + 0.2 * std_T) * marg[idx] * alphas[idx] / alphas_marg_sum
                    term_minus_std = term_minus_std + (mat_T - 0.2 * std_T) * marg[idx] * alphas[idx] / alphas_marg_sum

            if out_dim > 1:
                term_mean = term_mean[:, 0]
            mean_function[i,:] = term_mean

            if out_dim == 1:
                plus_std_function[i,:] = term_plus_std
                minus_std_function[i,:] = term_minus_std

            # # error for nmse
            # err = err + np.absolute((y_hat - mean_function[i]))
            # err_squared = err_squared + ((y_hat - mean_function[i]) ** 2)

        stop_prediction = timeit.default_timer()

        explained_var_test = explained_variance_score(data_test[:, in_dim_niw:], mean_function, multioutput='variance_weighted')

        # timer
        stop = timeit.default_timer()
        inf_time = start_plotting - start_inference
        plot_time = start_prediction - start_plotting
        pred_time = stop_prediction - start_prediction
        overall_time = stop - start

        # print('Inference time:', inf_time)
        # print('Plotting (training) time: ', plot_time)
        # print('Prediction time: ', pred_time)
        # print('Overall time: ', overall_time)

        # write to file
        f = open(csv_path, 'a+')
        f.write(str(explained_var_test) + " ")
        f.write(str(explained_var_train) + " ")
        f.write(str(vlb) + " ")
        f.write(str(len(dpglm.used_labels)) + " ")
        f.write(str(mf_iter) + " ")
        f.write(str(inf_time) + " ")
        f.write(str(pred_time))
        f.write("\n")
        f.close()

    # load data
    explained_var_test = pandas.read_csv(csv_path, header=None, dtype=None, engine='python', sep=" ", index_col=False, usecols=[0]).values  # skiprows=2

    data_intermed = explained_var_test  # np.column_stack((nmse_test, nMAE_test))

    # write statistics of data to stats file
    mean = np.mean(data_intermed, axis=0)
    std = np.std(data_intermed, axis=0)
    var = np.var(data_intermed, axis=0)
    median = np.median(data_intermed, axis=0)
    q1 = np.quantile(data_intermed, 0.25, axis=0)
    q3 = np.quantile(data_intermed, 0.75, axis=0)

    f = open(stat_path, 'w+')
    f.write('mean' + " " + str(mean) + "\n")
    f.write('std' + " " + str(std) + "\n")
    f.write('var' + " " + str(var) + "\n")
    f.write('median' + " " + str(median) + "\n")
    f.write('q1' + " " + str(q1) + "\n")
    f.write('q3' + " " + str(q3) + "\n")
    # f.write("\n")
    f.close()
