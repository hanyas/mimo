import numpy as np
import numpy.random as npr
from sklearn.metrics import explained_variance_score, r2_score
import pandas

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from mimo import distributions, models
from mimo.util.text import progprint_xrange
from mimo.util.data import *
from mimo.util.plot import *
from mimo.util.error_metrics import calc_error_metrics
from mimo.util.prediction import *
from mimo.util.mean_gibbs import calculate_mean_gibbs

import os
import copy
import operator
import random
import timeit
from io import StringIO
import datetime

# timer
start = timeit.default_timer()

# set random seed
seed = None
np.random.seed(seed=seed)
random.seed(seed)


dir = 'C:\\Users\\pistl\\Dropbox\\MA\\mimo_final\\'
CMB = True
SIN = False
kin_1joint = False
kin_2joint = False
kin_3joint = False
sarcos = False

n_train = 600

if CMB:
    x_categories = ['100', '200', '300', '400', '600']
    n_train_arr = [100,200,300,400,600]
    in_dim_niw = 1
    out_dim = 1
    str_dataset = 'CMB' #name of directory within evaluation/
    data_file = 'cmb.csv' #name of the file within datasets/

if SIN:
    x_categories = ['1000', '1500', '2000', '2500', '3000']
    n_train_arr = [1000, 1500, 2000, 2500, 3000]
    in_dim_niw = 1
    out_dim = 1
    str_dataset = 'SIN' #name of directory within evaluation/
    data_file = 'sin.csv' #name of the file within datasets/

if kin_1joint:
    x_categories = ['1000', '2000', '3000', '4000', '5000']
    n_train_arr = [1000, 2000, 3000, 4000, 5000]
    in_dim_niw = 1
    out_dim = 2
    str_dataset = 'kin_1joint' #name of directory within evaluation/
    data_file = 'kin_1joint.csv' #name of the file within datasets/

if kin_2joint:
    x_categories = ['1000', '2000', '3000', '4000', '5000']
    n_train_arr = [1000, 2000, 3000, 4000, 5000]
    in_dim_niw = 2
    out_dim = 2
    str_dataset = 'kin_2joint' #name of directory within evaluation/
    data_file = 'kin_2joint.csv' #name of the file within datasets/

if kin_3joint:
    x_categories = ['1000', '2000', '3000', '4000', '5000']
    n_train_arr = [1000, 2000, 3000, 4000, 5000]
    in_dim_niw = 3
    out_dim = 2
    str_dataset = 'kin_3joint' #name of directory within evaluation/
    data_file = 'kin_3joint.csv' #name of the file within datasets/

if sarcos:
    x_categories = ['1000', '2000', '3000', '4000', '5000']
    n_train_arr = [3000, 6000, 9000, 12000, 15000]
    in_dim_niw = 21
    out_dim = 7
    str_dataset = 'sarcos' #name of directory within evaluation/
    data_file = 'sarcos.csv' #name of the file within datasets/


n_test = int(n_train / 5)
data, data_test = load_data(n_train, n_test, data_file, dir, out_dim, in_dim_niw, sarcos)

# plt.plot(np.arange(1, n_train + 1), data[:, in_dim_niw], color="black")
# plt.show()
# plt.plot(np.arange(1, n_test + 1), data_test[:, in_dim_niw], color="black")
# plt.show()

# set working directory and file name
os.chdir(dir)
str_eval1 = ''

# general settings
affine = True
nb_models = 5
metaitr = 1
superitr = 1

# inference settings
gibbs = True
gibbs_iter = 300

mf_conv = True #mf with convergence criterion and max_iter

mf = False #mf with fixed iterations
mf_iter = 150

mf_sgd, batch_size, epochs, step_size = False, 30, 300, 1e-1

# gating settings
alpha_gatings = 1
stick_breaking = False

# plotting settings
legend_upper_right = True
plot_2d_prediction, save_2d_prediction = True, False

plot_kinematics, save_kinematics = False, False
plot_dynamics, save_dynamics = False, False

plot_vlb = True
# plot_metrics = False # only for mf with fixed iterations


# set pathes
# header1 = str(datetime.datetime.now().strftime('date, time: %Y<f-%m-%d, %H-%M-%S'))
# header2 = "1:nMSE_train 2:nMAE_train 3:VLB 4:nMSE_test 5:nMAE_test 6:used_labels 7:mf_iter 8:gibbs_iter 9:inf_time 10:pred_time"

csv_path = os.path.join('evaluation/' + str(str_dataset) + '/raw/' + str(str_dataset) + '_' + str(str_eval1) + '_raw' + '.csv')
tikz_path = os.path.join('evaluation/' + str(str_dataset) + '/tikz/' + str(str_dataset) + '_' + str(str_eval1)  + '.tex')
pdf_path = os.path.join('evaluation/' + str(str_dataset) + '/pdf/' + str(str_dataset) + '_' + str(str_eval1)  + '.pdf')
stat_path = os.path.join('evaluation/' + str(str_dataset) + '/stats/' + str(str_dataset) + '_' + str(str_eval1) +  '_stats' + '.csv')

visual_tikz_path = os.path.join('evaluation/' + str(str_dataset) + '/visual/' + str(str_dataset) + '_' + str(str_eval1))
visual_pdf_path = os.path.join('evaluation/' + str(str_dataset) + '/visual/' + str(str_dataset) + '_' + str(str_eval1))

# # write headers
# f = open(csv_path, 'a+')
# f.write(header1 + "\n")
# f.write(header2 + "\n")
# f.close()



# data, data_test = generate_SIN(n_train,in_dim_niw, out_dim, freq=3, shuffle=False, seed=seed), generate_SIN(n_test,in_dim_niw, out_dim, freq=3, shuffle=False, seed=seed)
# data, data_test = generate_CMB(n_train, n_test,seed)
# data, data_test = generate_LIN(n_train,in_dim_niw, out_dim, freq=14, shuffle=False, seed=seed),generate_LIN(n_test,in_dim_niw, out_dim, freq=14, shuffle=False, seed=seed)
# data, data_test = generate_gaussian(n_train, out_dim, in_dim_niw, seed=seed), generate_gaussian(n_test, out_dim, in_dim_niw, seed=seed)
# data, data_test = generate_heaviside2(n_train,scaling=1), generate_heaviside2(n_test,scaling=1)
# data, data_test = generate_kinematics(n_train=n_train,out_dim=out_dim,num_joints=in_dim_niw,loc_noise=0,scale_noise=5,l1=1,l2=1,l3=1,l4=1,seed=seed), generate_kinematics(n_train=n_test,out_dim=out_dim,num_joints=in_dim_niw,loc_noise=0,scale_noise=5,l1=1,l2=1,l3=1,l4=1,seed=seed)

# data, data_test = generate_Sarcos(n_train, n_test, in_dim_niw, out_dim, seed, all=True, flag=False),generate_Sarcos(n_train, n_test, in_dim_niw, out_dim, seed, all=True, flag=True)
# data, data_test = generate_Barret(n_train, None, in_dim_niw, out_dim, seed, all=True), generate_Barret(n_train, None, in_dim_niw, out_dim, seed, all=True)
# data, data_test = normalize_data(data, 1),normalize_data(data_test,1)





# define gating
if stick_breaking == False:
    gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, )) * alpha_gatings)
    gating_prior = distributions.Dirichlet(**gating_hypparams)
else:
    gating_hypparams = dict(K=nb_models, gammas=np.ones((nb_models, )), deltas=np.ones((nb_models, )) * alpha_gatings)
    gating_prior = distributions.StickBreaking(**gating_hypparams)

for m in range(metaitr):

    # initialize prior parameters: draw from uniform disitributions
    components_prior = []

    in_dim_mniw = in_dim_niw
    if affine:
        in_dim_mniw = in_dim_niw + 1

    V = np.eye(in_dim_mniw)
    for i in range(in_dim_mniw):
        if i < in_dim_mniw - 1:
            V[i, i] = npr.uniform(0, 10)
        else:
            V[i, i] = npr.uniform(0, 1000) # offset
    rnd_psi_mniw = npr.uniform(0, 0.1)
    nu_mniw = 2 * out_dim + 1

    mu_low = np.amin(data[:,:-out_dim])
    mu_high = np.amax(data[:,:-out_dim])

    rnd_psi_niw = npr.uniform(0, 10)
    rnd_kappa = npr.uniform(0,0.1)

    for j in range(nb_models):
        components_hypparams = dict(mu=npr.uniform(mu_low,mu_high,size=in_dim_niw),
                                    kappa= rnd_kappa,  #0.05
                                    psi_niw=np.eye(in_dim_niw) * rnd_psi_niw,
                                    nu_niw=2 * in_dim_niw + 1,
                                    M=np.zeros((out_dim, in_dim_mniw)),
                                    V=V,
                                    affine=affine,
                                    psi_mniw=np.eye(out_dim) * rnd_psi_mniw,
                                    nu_mniw= nu_mniw )
        components_prior_rand = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)
        components_prior.append(components_prior_rand)

    # define model
    if stick_breaking == False:
        dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                             components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(nb_models)])
    else:
        dpglm = models.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                              components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(nb_models)])

    dpglm.add_data(data)

    start_inference = timeit.default_timer()

    # inference
    allscores = []
    allmodels = []
    all_nMSE = []
    all_err = []
    for _ in range(superitr):
        scores = []
        nMSE = []
        err_ = []

        # Gibbs sampling to wander around the posterior
        if gibbs == True:
            print('Gibbs Sampling')
            for _ in progprint_xrange(gibbs_iter):
            # for _ in range(gibbs_iter):
                dpglm.resample_model()
                # this is needed if Gibbs sampling is used without Mean Field
                if mf != True and mf_conv != True and mf_sgd != True:
                    for idx, l in enumerate(dpglm.labels_list):
                        l.r = l.get_responsibility()
                    scores.append(dpglm._vlb())

        if mf == True:
            # mean field to lock onto a mode (without convergence criterion, with fixed number of iterations)
            print('Mean Field')
            if gibbs != True:
                dpglm.resample_model()
            # scores = [dpglm.meanfield_coordinate_descent_step() for _ in progprint_xrange(250)]
            # for _ in progprint_xrange(mf_iter):
            for _ in range(mf_iter):
                scores.append(dpglm.meanfield_coordinate_descent_step())

                # pred_y, err, err_squared = predict_train(dpglm, data, out_dim, in_dim_niw)
                # var = np.var(data[:, in_dim_niw:], axis=0)
                # MSE = 1 / n_train * err_squared
                # nMSE.append(np.sum(MSE[0] / var[0]))
                # err_.append(np.sum(err))

        if mf_conv == True:
            # mean field to lock onto a mode (with convergence criterion)
            print('Mean Field')
            if gibbs != True:
                dpglm.resample_model()
            scores = dpglm.meanfield_coordinate_descent(tol=1e-2, maxiter=500, progprint=True)

        # stochastic mean field to lock onto a mode
        if mf_sgd == True:
            print('Stochastic Mean Field')
            if gibbs != True:
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
            # allscores.append(dpglm.meanfield_coordinate_descent_step())

        # all_err.append(err_)
        # all_nMSE.append(nMSE)
        allscores.append(scores)
        allmodels.append(copy.deepcopy(dpglm))



    start_plotting = timeit.default_timer()

    # plot vlb and error metrics over iterations
    if plot_vlb:
        plot_scores(allscores)
    # if mf and plot_metrics:
    #     plot_nMSE(all_nMSE)
    #     plot_absolute_error(all_err)

    # Sort models and select best one
    models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)],
                               key=operator.itemgetter(1), reverse=True)
    dpglm = models_and_scores[0][0]
    vlb = models_and_scores[0][1]

    # get labels
    for l in dpglm.labels_list:
        label = l.z
    # print('used labels',dpglm.used_labels)
    print('# used labels',len(dpglm.used_labels))

    # predict on training data
    pred_y, err, err_squared = predict_train(dpglm, data, out_dim, in_dim_niw)

    # calculate error metrics on training data
    # nMSE_train = calc_error_metrics(data, n_train, in_dim_niw, err, err_squared)[0]
    # nMAE_train = calc_error_metrics(data, n_train, in_dim_niw, err, err_squared)[1]
    explained_var_train = explained_variance_score(data[:, in_dim_niw:], pred_y,multioutput='variance_weighted')
    print(explained_var_train)

    # plots of 2d prediction for training data
    if plot_2d_prediction:
        if in_dim_niw + out_dim == 2:
            plot_prediction_2d(data, pred_y, 'Training Data', save_2d_prediction, visual_pdf_path, visual_tikz_path, legend_upper_right)

    # plot of prediction for endeffector positions vs. data
    if plot_kinematics == True:
        endeffector_pos_2d(data, in_dim_niw, pred_y, 'Training Data', visual_pdf_path, visual_tikz_path, save_kinematics)
        if in_dim_niw == 1:
         # 3d plot of x,y position endeffector and angle
            endeffector_pos_3d(data, pred_y, in_dim_niw, 'Training Data', visual_pdf_path, visual_tikz_path, save_kinematics)

    # plot of inverse dynamics of first joint: motor torque and predicted motor torque
    if plot_dynamics == True:
        motor_torque(n_train, data, pred_y, in_dim_niw, save_dynamics, visual_tikz_path, visual_pdf_path,
                     'Training Data')

    start_prediction = timeit.default_timer()


    calculate_mean_gibbs(dpglm, data_test, n_test, 5, dir)


    # # calculate alpha_hat (sum over alpha_k)
    # if not stick_breaking:
    #     alphas_hat = 0
    #     alphas = dpglm.gating.posterior.alphas
    #     for idx, c in enumerate(dpglm.components):
    #         if idx in dpglm.used_labels:
    #             alphas_hat = alphas_hat + alphas[idx]
    # else:
    #     stick_lengths = np.ones([len(dpglm.components)])
    #     product = np.ones([len(dpglm.components)])
    #     gammas = dpglm.gating.posterior.gammas
    #     deltas = dpglm.gating.posterior.deltas
    #     for idx, c in enumerate(dpglm.components):
    #         if idx in dpglm.used_labels:
    #             product[idx] = gammas[idx] / gammas[idx] + deltas[idx]
    #             for j in range(idx):
    #                 product[idx] = product[idx] * gammas[j] / gammas[j] + deltas[j]
    #     alphas = product
    #
    #
    # # initialize variables
    # mean_function, plus_std_function, minus_std_function = np.empty_like(data_test[:,in_dim_niw:]), np.empty_like(data_test[:,in_dim_niw:]), np.empty_like(data_test[:,in_dim_niw:])
    # marg, stud_t = np.zeros([len(dpglm.components)]), np.zeros([len(dpglm.components)])
    # err, err_squared = 0, 0
    # dot_xx = np.zeros([len(dpglm.components), in_dim_mniw, in_dim_mniw])
    # dot_yx = np.zeros([len(dpglm.components), out_dim, in_dim_mniw])
    # dot_yy = np.zeros([len(dpglm.components), out_dim, out_dim])
    # n = np.zeros([len(dpglm.components)])
    #
    # # get statistics for each component for training data
    # for idx, c in enumerate(dpglm.components):
    #
    #     statistics = c.posterior.get_statistics([l.data[l.z == idx] for l in dpglm.labels_list])
    #     dot_yx[idx], dot_xx[idx], dot_yy[idx], n[idx] = statistics[4], statistics[5], statistics[6], statistics[7]
    #
    # # prediction / mean function of y_hat for all training data x_hat
    # for i in range(len(data_test[:, :-out_dim])):
    #     x_hat = data_test[i, :-out_dim]
    #     y_hat = data_test[i, in_dim_niw:]
    #
    #     # calculate the marginal likelihood of training data x_hat for each cluster (under NIW distribution)
    #     # calculate the normalization term for mean function (= alphas_marg_sum) for x_hat
    #     alphas_marg_sum = 0
    #     for idx, c in enumerate(dpglm.components):
    #         if idx in dpglm.used_labels:
    #             mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw = get_component_standard_parameters(c.posterior)
    #             marg[idx] = NIW_marg_likelihood(x_hat, mu, kappa, psi_niw, nu_niw, 1, out_dim)
    #             alphas_marg_sum = alphas_marg_sum + alphas[idx] * marg[idx]
    #
    #     # calculate contribution of each cluster to mean function / prediction for training data x_hat
    #     term_mean = 0
    #     term_plus_std = 0
    #     term_minus_std = 0
    #     for idx, c in enumerate(dpglm.components):
    #         if idx in dpglm.used_labels:
    #             mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw = get_component_standard_parameters(c.posterior)
    #             S_0, N_0 = get_component_standard_parameters(c.prior)[6], get_component_standard_parameters(c.prior)[7]
    #             mat_T, std_T = matrix_t(data,idx,label,out_dim,in_dim_niw,affine,x_hat, V, M, nb_models, S_0, dot_xx[idx], dot_yx[idx], dot_yy[idx], psi_mniw, nu_mniw, N_0)
    #
    #             # mean function
    #             term_mean = term_mean + mat_T * marg[idx] * alphas[idx] / alphas_marg_sum
    #             # term_mean = term_mean + mat_T * marg[idx] / marg.sum()
    #             # term = term + mat_T * alphas[idx] * marg[idx]  / alphas_sum * len(dpglm.components)
    #
    #             # confidence interval
    #             term_plus_std = term_plus_std + (mat_T + 2 * std_T) * marg[idx] * alphas[idx] / alphas_marg_sum
    #             term_minus_std = term_minus_std + (mat_T - 2 * std_T) * marg[idx] * alphas[idx] / alphas_marg_sum
    #
    #     if out_dim > 1:
    #         term_mean = term_mean[:, 0]
    #     mean_function[i,:] = term_mean
    #
    #     if out_dim == 1:
    #         plus_std_function[i,:] = term_plus_std
    #         minus_std_function[i,:] = term_minus_std
    #
    #     # # error for nMSE
    #     # err = err + np.absolute((y_hat - mean_function[i]))
    #     # err_squared = err_squared + ((y_hat - mean_function[i]) ** 2)
    #
    # stop_prediction = timeit.default_timer()
    #
    # if plot_2d_prediction:
    #     if in_dim_niw + out_dim == 2:
    #         # plot_prediction_2d(data_test, mean_function)
    #         plot_prediction_2d(data_test, mean_function, 'Test Data', save_2d_prediction, visual_pdf_path, visual_tikz_path,
    #                            legend_upper_right)
    #         plot_prediction_2d_mean(data_test, mean_function, plus_std_function, minus_std_function, 'Test Data', save_2d_prediction, visual_pdf_path, visual_tikz_path, legend_upper_right)
    #
    # # plot of kinematics data
    # if plot_kinematics == True:
    #     # plot of prediction for endeffector positions vs. data
    #     endeffector_pos_2d(data_test, in_dim_niw, mean_function, 'Test Data', visual_pdf_path, visual_tikz_path, save_kinematics)
    #     if in_dim_niw == 1:
    #         # 3d plot of x,y position endeffector and angle
    #         endeffector_pos_3d(data_test, mean_function, in_dim_niw, 'Test Data', visual_pdf_path, visual_tikz_path, save_kinematics)
    #
    # # plot of inverse dynamics of first joint: motor torque and predicted motor torque
    # if plot_dynamics == True:
    #     motor_torque(n_test, data_test, mean_function, in_dim_niw, save_dynamics, visual_tikz_path, visual_pdf_path, 'Test Data')
    #
    # # nMSE_test = calc_error_metrics(data_test, n_test, in_dim_niw, err, err_squared)[0]
    # # nMAE_test = calc_error_metrics(data_test, n_test, in_dim_niw, err, err_squared)[1]
    # explained_var_test= explained_variance_score(data_test[:, in_dim_niw:], mean_function,multioutput='variance_weighted')
    # print(explained_var_test)
    #
    # # timer
    # stop = timeit.default_timer()
    # inf_time = start_plotting - start_inference
    # plot_time = start_prediction - start_plotting
    # pred_time = stop_prediction - start_prediction
    # overall_time = stop - start
    #
    # print('Inference time:', inf_time)
    # print('Plotting (training) time: ', plot_time)
    # print('Prediction time: ', pred_time)
    # print('Overall time: ', overall_time)

#     # write to file
#     f = open(csv_path, 'a+')
#     f.write(str(nMSE_train[0]) + " ")
#     f.write(str(nMAE_train[0])+ " ")
#     f.write(str(vlb) + " ")
#     f.write(str(nMSE_test[0]) + " ")
#     f.write(str(nMAE_test[0]) + " ")
#     f.write(str(len(dpglm.used_labels)) + " ")
#     f.write(str(mf_iter) + " ")
#     f.write(str(gibbs_iter) + " ")
#     f.write(str(inf_time) + " ")
#     f.write(str(pred_time))
#     f.write("\n")
#     f.close()
#
# # load data
# # nMSE_train = pandas.read_csv(csv_path,header=None,dtype = None,engine='python', sep=" ",index_col=False,usecols=[0]).values
# # nMAE_train = pandas.read_csv(csv_path,header=None,dtype = None,engine='python', sep=" ",index_col=False,usecols=[1]).values
# # vlb = pandas.read_csv(csv_path,header=None,dtype = None,engine='python', sep=" ",index_col=False,usecols=[2]).values
# nMSE_test = pandas.read_csv(csv_path, header=None, dtype=None, engine='python', sep=" ", index_col=False, usecols=[3], skiprows=2).values
# nMAE_test = pandas.read_csv(csv_path, header=None, dtype=None, engine='python', sep=" ", index_col=False, usecols=[4], skiprows=2).values
# # len(dpglm.used_labels) = pandas.read_csv(csv_path,header=None,dtype = None,engine='python', sep=" ",index_col=False,usecols=[5]).values
# # mf_iter = pandas.read_csv(csv_path,header=None,dtype = None,engine='python', sep=" ",index_col=False,usecols=[6]).values
# # gibbs_iter = pandas.read_csv(csv_path,header=None,dtype = None,engine='python', sep=" ",index_col=False,usecols=[7]).values
# # inf_time = pandas.read_csv(csv_path,header=None,dtype = None,engine='python', sep=" ",index_col=False,usecols=[8]).values
# # pred_time = pandas.read_csv(csv_path,header=None,dtype = None,engine='python', sep=" ",index_col=False,usecols=[9]).values
# # data_violin= np.genfromtxt(csv_path,delimiter=",")
# # data_violin = np.loadtxt(csv_path.spli, skiprows=2, usecols=(3,4), delimiter=",") # dtype=None, encoding=None
#
#
# # generate and save violin plots
# num_cols = 2
# data_violin = np.column_stack((nMSE_test, nMAE_test))
# x_label = 'Number of training data points'
# x_categories = ['500', '1000']
# y_label = 'nMSE'
# title = 'Violin plots for nMSE of CMB dataset'
#
# violin_plot(data_violin, num_cols, tikz_path, pdf_path, x_label, y_label, title, x_categories)
#
# # write statistics of data to stats file
# mean = np.mean(data_violin, axis=1)
# std = np.std(data_violin, axis=1)
# var = np.var(data_violin, axis=1)
# median = np.median(data_violin, axis=1)
# q1 = np.quantile(data_violin, 0.25, axis=1)
# q3 = np.quantile(data_violin, 0.75, axis=1)
#
# f = open(stat_path, 'w+')
# f.write('mean'+ " " + str(mean) + "\n")
# f.write('std'+ " " + str(std) + "\n")
# f.write('var'+ " " + str(var) + "\n")
# f.write('median'+ " " + str(median) + "\n")
# f.write('q1'+ " " + str(q1) + "\n")
# f.write('q3' + " "+ str(q3)+ "\n")
# # f.write("\n")
# f.close()
#
# # #remove file
# # os.remove(csv_path)

