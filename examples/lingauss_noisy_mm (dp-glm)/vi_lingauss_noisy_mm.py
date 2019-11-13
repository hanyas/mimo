import numpy as np
import numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import datetime

from mimo import distributions, models
from mimo.util.text import progprint_xrange
from mimo.util.data import generate_CMB, generate_SIN, generate_kinematics, generate_heaviside1, generate_heaviside2, generate_gaussian, generate_Sarcos, generate_Barret,generate_kinematics_cos, generate_LIN, center_data, normalize_data
from mimo.util.plot import plot_absolute_error, plot_scores, plot_nMSE, plot_prediction_2d, endeffector_pos_2d, endeffector_pos_3d, motor_torque, plot_prediction_2d_mean, violin_plot
from mimo.util.error_metrics import calc_error_metrics
from mimo.util.prediction import *

import os
import sys
import copy
import operator
import random
import timeit


# timer
start = timeit.default_timer()
# set random seed
seed = None
np.random.seed(seed=seed)
random.seed(seed)
# set boolean flags to standard values
plot_kinematics = False
plot_dynamics = False


# set working directory
# dirname = os.path.dirname(vi_lingauss_noisy_mm.py)
os.chdir('C:\\Users\\pistl\\Dropbox\\MA\\mimo_final\\')
str_dataset = 'CMB'
str_evaluated = 'components'

txt_path = os.path.join('evaluation/' + str(str_dataset) + '/files/' + str(str_dataset) + '_' + str(str_evaluated) + '.txt')
tikz_path = os.path.join('evaluation/' + str(str_dataset) + '/pdfs/' + str(str_dataset) + '_' + str(str_evaluated) + '.ticz')
pdf_path = os.path.join('evaluation/' + str(str_dataset) + '/tikz/' + str(str_dataset) + '_' + str(str_evaluated) + '.pdf')

# write headers
# f = open(txt_path, 'a+')
# f.write(str(datetime.datetime.now().strftime('date, time: %Y-%m-%d, %H-%M-%S')) + "\n")
# f.write("1:nMSE 2:nMAE 3:VLB 4:used_labels 5:mf_iter 6:gibbs_iter 7:inf_time 8:pred_time"+ "\n")
# f.close()

# general settings
affine = True
nb_models = 15
metaitr = 1
superitr = 1

# inference settings
gibbs = True
gibbs_iter = 150

mf_conv = True #mf with convergence criterion and max_iter

mf = False #mf with fixed iterations
mf_iter = 150

mf_sgd, batch_size, epochs, step_size = False, 30, 300, 1e-1

# generate data
in_dim_niw = 1
out_dim = 1
n_train = 500
n_test = 300
freq = 14 #frequencz for sine data

# gating settings
scale_gating = 1
stick_breaking = False

# plotting settings
plot_vlb = False
plot_training = False
plot_prediction = False



# choose which data to generate
# main: sin, cmb, kinematics, dynamics_sarcos, dynamics_barrett
# others: gaussian, heaviside

# data, data_test = generate_SIN(n_train,in_dim_niw, out_dim, freq=freq, shuffle=False, seed=seed), generate_SIN(n_test,in_dim_niw, out_dim, freq=freq, shuffle=False, seed=seed)
data, data_test = generate_CMB(n_train, seed), generate_CMB(n_test, seed)
# data, data_test = generate_LIN(n_train,in_dim_niw, out_dim, freq=14, shuffle=False, seed=seed),generate_LIN(n_test,in_dim_niw, out_dim, freq=14, shuffle=False, seed=seed)
# data, data_test = generate_gaussian(n_train, out_dim, in_dim_niw, seed=seed), generate_gaussian(n_test, out_dim, in_dim_niw, seed=seed)
# data, data_test = generate_heaviside2(n_train,scaling=1), generate_heaviside2(n_test,scaling=1)

# data, data_test, plot_kinematics = generate_kinematics(n_train=n_train,out_dim=out_dim,num_joints=in_dim_niw,loc_noise=0,scale_noise=5,l1=1,l2=1,l3=1,l4=1,seed=seed), generate_kinematics(n_train=n_test,out_dim=out_dim,num_joints=in_dim_niw,loc_noise=0,scale_noise=5,l1=1,l2=1,l3=1,l4=1,seed=seed), True

# data, data_test, plot_dynamics = generate_Sarcos(n_train, None, in_dim_niw, out_dim, seed, all=True),generate_Sarcos(n_test, None, in_dim_niw, out_dim, seed, all=True), True
# data, data_test, plot_dynamics = generate_Barret(n_train, None, in_dim_niw, out_dim, seed, all=True), generate_Barret(n_train, None, in_dim_niw, out_dim, seed, all=True), True

data, data_test = normalize_data(data, 1),normalize_data(data_test,1)



# define gating
if stick_breaking == False:
    gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, ))*scale_gating)
    gating_prior = distributions.Dirichlet(**gating_hypparams)
else:
    gating_hypparams = dict(K=nb_models, gammas=np.ones((nb_models, )), deltas=np.ones((nb_models, ))*scale_gating)
    gating_prior = distributions.StickBreaking(**gating_hypparams)


for m in range(metaitr):
    # define components
    components_prior = []
    for j in range(nb_models):
        if affine:
            in_dim_mniw = in_dim_niw + 1
        else:
            in_dim_mniw = in_dim_niw

        V = np.eye(in_dim_mniw)
        for i in range(in_dim_mniw):
            if i < in_dim_mniw - 1:
                V[i, i] = npr.uniform(0.1, 10)
            else:
                V[i, i] = npr.uniform(1, 1000) # offset
        rnd_psi_mniw = npr.uniform(0.001, 0.1)
        nu_mniw = 2 * out_dim + 1

        mu_low = np.amin(data[:,:-out_dim])
        mu_high = np.amax(data[:,:-out_dim])

        rnd_psi_niw = npr.uniform(0.1, 10)
        rnd_kappa = npr.uniform(0.01,0.1)

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
                dpglm.resample_model()
                if mf != True and mf_conv != True and mf_sgd != True:
                    # this part is needed if Gibbs sampling is used without Mean Field
                    for idx, l in enumerate(dpglm.labels_list):
                        l.r = l.get_responsibility()
                    scores.append(dpglm._vlb())

        if mf == True:
            # mean field to lock onto a mode (without convergence criterion)
            print('Mean Field')
            if gibbs != True:
                dpglm.resample_model()
            # scores = [dpglm.meanfield_coordinate_descent_step() for _ in progprint_xrange(250)]
            for _ in progprint_xrange(mf_iter):
                scores.append(dpglm.meanfield_coordinate_descent_step())

                pred_y, err, err_squared = predict_train(dpglm, data, out_dim, in_dim_niw)
                var = np.var(data[:, in_dim_niw:], axis=0)
                MSE = 1 / n_train * err_squared
                nMSE.append(np.sum(MSE[0] / var[0]))
                err_.append(np.sum(err))

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
            for _ in progprint_xrange(epochs):
                minibatch = npr.permutation(n_train)[:minibatchsize]
                dpglm.meanfield_sgdstep(minibatch=data[minibatch], prob=prob, stepsize=step_size)
                for idx, l in enumerate(dpglm.labels_list):
                    l.r = l.get_responsibility()
                scores.append(dpglm._vlb())
            # allscores.append(dpglm.meanfield_coordinate_descent_step())

        all_err.append(err_)
        all_nMSE.append(nMSE)
        allscores.append(scores)
        allmodels.append(copy.deepcopy(dpglm))



    start_plotting = timeit.default_timer()

    # plot error metrics over iterations
    if plot_vlb:
        plot_scores(allscores)
    if mf == True:
        plot_nMSE(all_nMSE)
        plot_absolute_error(all_err)

    # Sort models and select best one
    models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)],
                               key=operator.itemgetter(1), reverse=True)
    dpglm = models_and_scores[0][0]
    vlb = models_and_scores[0][1]

    # print labels
    for l in dpglm.labels_list:
        label = l.z
    # print('used labels',dpglm.used_labels)
    # print('# used labels',len(dpglm.used_labels))

    if plot_training:
        # predict on training data
        pred_y, err, err_squared = predict_train(dpglm, data, out_dim, in_dim_niw)

        # calculate and print error metrics (MAE, nMAE, MSE, nMSE, RMSE, nRMSE)
        calc_error_metrics(data, n_train, in_dim_niw, err, err_squared)

        # plots of prediction for training data
        if in_dim_niw + out_dim == 2:
            plot_prediction_2d(data, pred_y)
        if plot_kinematics == True:
            # plot of prediction for endeffector positions vs. data
            endeffector_pos_2d(data, in_dim_niw, pred_y, 'results/kin_train.pdf')
            if in_dim_niw == 1:
                # 3d plot of x,y position endeffector and angle
                endeffector_pos_3d(data, pred_y, in_dim_niw, 'wasd')
        if plot_dynamics == True:
            # plot of inverse dynamics of first joint: motor torque and predicted motor torque
            motor_torque(n_train, data, pred_y, in_dim_niw)



    start_prediction = timeit.default_timer()

    if stick_breaking == False:
        alphas_hat = 0
        alphas = dpglm.gating.posterior.alphas
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                alphas_hat = alphas_hat + alphas[idx]

    mean_function, plus_std_function, minus_std_function = np.empty_like(data_test[:,in_dim_niw:]), np.empty_like(data_test[:,in_dim_niw:]), np.empty_like(data_test[:,in_dim_niw:])
    marg, stud_t = np.zeros([len(dpglm.components)]), np.zeros([len(dpglm.components)])
    err, err_squared = 0, 0

    dot_xx = np.zeros([len(dpglm.components), in_dim_mniw, in_dim_mniw])
    dot_yx = np.zeros([len(dpglm.components), out_dim, in_dim_mniw])
    dot_yy = np.zeros([len(dpglm.components), out_dim, out_dim])
    n = np.zeros([len(dpglm.components)])

    # for l in dpglm.labels_list:
        # print('lendata1',len(l.data))

    for idx, c in enumerate(dpglm.components):

        statistics = c.posterior.get_statistics([l.data[l.z == idx] for l in dpglm.labels_list])
        dot_yx[idx], dot_xx[idx], dot_yy[idx], n[idx] = statistics[4], statistics[5], statistics[6], statistics[7]

        # if idx in dpglm.used_labels:
            # dot_yxT, dot_xxT, dot_yyT = c.posterior.get
            # data_cluster = [l.data[l.z == idx] for l in dpglm.labels_list]
            # print('lendata2',len(data_cluster))
    #         # print('data_cluster', data_cluster)
    #         # print('data_cluster[0]', data_cluster[0])
    #         # x = data_cluster[0][:, :-out_dim]
    #         # y = data_cluster[0][:, in_dim_niw:]
    #         # dot_xx[idx] = np.dot(x, x.T)
    #         # print('x', x)
    #         # print('y', y)
    #         # print(x.shape, y.shape)
    #         # print(np.dot(y, x.T))
    #         # dot_yx[idx] = np.dot(y, x.T)
    #         # dot_yy[idx] = np.dot(y, y.T)

    for i in range(len(data_test[:, :-out_dim])):
        x_hat = data_test[i, :-out_dim]
        y_hat = data_test[i, in_dim_niw:]

        alphas_marg_sum = 0
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw = get_component_standard_parameters(c.posterior)
                marg[idx] = NIW_marg_likelihood(x_hat, mu, kappa, psi_niw, nu_niw, 1, out_dim)
                # stud_t[idx] = student_t(x_hat, mu, kappa, psi_niw, nu_niw, in_dim_niw)
                if np.isinf(marg[idx] * alphas[idx]):
                    alphas_marg_sum = alphas_marg_sum
                else:
                    alphas_marg_sum = alphas_marg_sum + alphas[idx] * marg[idx]
                # print(alphas_marg_sum, marg[idx], alphas[idx])
        # print('-------------------')
        term_mean = 0
        term_plus_std = 0
        term_minus_std = 0
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw = get_component_standard_parameters(c.posterior)
                S_0, N_0 = get_component_standard_parameters(c.prior)[6], get_component_standard_parameters(c.prior)[7]

                # mat_T, std_T =1,1
                mat_T, std_T = matrix_t(data,idx,label,out_dim,in_dim_niw,affine,x_hat, V, M, nb_models, S_0, dot_xx[idx], dot_yx[idx], dot_yy[idx], psi_mniw)

                # mean function
                term_mean = term_mean + mat_T * marg[idx] * alphas[idx] / alphas_marg_sum
                # term_mean = term_mean + mat_T * marg[idx] / marg.sum()
                # term = term + mat_T #* mu #* alphas[idx]
                # term = term + mat_T * alphas[idx] * marg[idx]  / alphas_marg_sum * len(dpglm.components)
                # term = term + mat_T * stud_t[idx] / stud_t.sum() * alphas[idx] * len(dpglm.used_labels)
                # term = term + mat_T * stud_t[idx] / marg[idx] * alphas[idx]

                # 90% confidence interval
                # term_plus_std = term_plus_std + (mat_T + 0.1645 * std_T) * marg[idx] / marg.sum()
                # term_minus_std = term_minus_std + (mat_T - 0.1645 * std_T) * marg[idx] / marg.sum()
                term_plus_std = term_plus_std + (mat_T + 0.2 * std_T) * marg[idx] * alphas[idx] / alphas_marg_sum
                term_minus_std = term_minus_std + (mat_T - 0.2 * std_T) * marg[idx] * alphas[idx] / alphas_marg_sum

        if out_dim > 1:
            term_mean = term_mean[:, 0]
        mean_function[i,:] = term_mean #/ alphas_hat

        if out_dim == 1:
            plus_std_function[i,:] = term_plus_std
            minus_std_function[i,:] = term_minus_std

        err = err + np.absolute((y_hat - mean_function[i]))
        err_squared = err_squared + ((y_hat - mean_function[i]) ** 2)

    stop_prediction = timeit.default_timer()


    if plot_prediction:
        if in_dim_niw + out_dim == 2:
            # plot_prediction_2d(data_test, mean_function)
            plot_prediction_2d_mean(data_test, mean_function, plus_std_function, minus_std_function)

        # plot of kinematics data
        if plot_kinematics == True:
            # plot of prediction for endeffector positions vs. data
            endeffector_pos_2d(data_test, in_dim_niw, mean_function, 'results/kin_test.pdf')
            if in_dim_niw == 1:
                # 3d plot of x,y position endeffector and angle
                endeffector_pos_3d(data_test, mean_function, in_dim_niw, 'wasd')
        if plot_dynamics == True:
            # plot of inverse dynamics of first joint: motor torque and predicted motor torque
            motor_torque(n_test, data_test, mean_function, in_dim_niw)

    nMSE = calc_error_metrics(data_test, n_test, in_dim_niw, err, err_squared)[0]
    nMAE = calc_error_metrics(data_test, n_test, in_dim_niw, err, err_squared)[1]

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


    # write nMSE for each iteration
    f = open(txt_path, 'a+')
    f.write(str(nMSE[0]) + " ")
    f.write(str(nMAE[0]) + " ")
    f.write(str(vlb) + " ")
    f.write(str(len(dpglm.used_labels)) + " ")
    f.write(str(mf_iter) + " ")
    f.write(str(gibbs_iter) + " ")
    f.write(str(inf_time) + " ")
    f.write(str(pred_time) + " ")
    f.write("\n")
    f.close()

data_violin = np.genfromtxt(txt_path, dtype=None, encoding=None, usecols=(0))

# preprocess data
# data_violin = np.delete(data_violin, [0,1], axis=0) # delete headers (first two rows)
data_violin = data_violin[np.logical_not(np.isnan(data_violin))]
data_violin = data_violin[np.logical_not(np.isinf(data_violin))]

violin_plot(data_violin, num_columns=1, tikz_path=tikz_path, pdf_path=pdf_path)





