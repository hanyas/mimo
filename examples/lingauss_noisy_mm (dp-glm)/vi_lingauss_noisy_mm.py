import numpy as np
import numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from mimo import distributions, models
from mimo.util.text import progprint_xrange
from mimo.util.data import generate_CMB, generate_SIN, generate_kinematics, generate_heaviside1, generate_heaviside2, generate_gaussian, generate_Sarcos, generate_Barret,generate_kinematics_cos, generate_LIN, center_data, normalize_data
from mimo.util.plot import plot_absolute_error, plot_scores, plot_nMSE, plot_prediction_2d, endeffector_pos, motor_torque, plot_prediction_2d_mean
from mimo.util.error_metrics import calc_error_metrics
from mimo.util.prediction import *

import sys
import copy
import operator
import random
import timeit

# for eachArg in sys.argv:
#         print(eachArg)

# timer
start = timeit.default_timer()
# set random seed
seed = None
np.random.seed(seed=seed)
random.seed(seed)
# set boolean flags to standard values
plot_kinematics = False
plot_dynamics = False



# setting
affine = True
nb_models = 25
superitr = 1
metaitr = 1
stick_breaking = False

plot_vlb = True
plot_training = True
plot_prediction = True

# set inference methods
gibbs = True
gibbs_iter = 150

mf = False #mf with fixed iterations
mf_iter = 150

mf_conv = True #mf with convergence criterion

mf_sgd = False
batch_size = 30
epochs = 300
step_size = 1e-1




# generate data
in_dim_niw = 1
out_dim = 1
n_train = 800
n_test = 300
freq = 14

# choose which data to generate and set plotting flags
# main: sin, cmb, kinematics, dynamics_sarcos, dynamics_barrett
# others: gaussian, heaviside

# data, data_test = generate_SIN(n_train,in_dim_niw, out_dim, freq=freq, shuffle=False, seed=seed), generate_SIN(n_test,in_dim_niw, out_dim, freq=freq, shuffle=False, seed=seed)
data, data_test = generate_CMB(n_train, seed), generate_CMB(n_test, seed)
# data, data_test = generate_LIN(n_train,in_dim_niw, out_dim, freq=14, shuffle=False, seed=seed),generate_LIN(n_test,in_dim_niw, out_dim, freq=14, shuffle=False, seed=seed)
# data, data_test = generate_gaussian(n_train, out_dim, in_dim_niw), generate_gaussian(n_test, out_dim, in_dim_niw)
# data, data_test = generate_heaviside2(n_train,scaling=1), generate_heaviside2(n_test,scaling=1)

# data, data_test, plot_kinematics = generate_kinematics(n_train=n_train,out_dim=out_dim,num_joints=in_dim_niw,loc_noise=0,scale_noise=5,l1=1,l2=1,l3=1), generate_kinematics(n_train=n_test,out_dim=out_dim,num_joints=in_dim_niw,loc_noise=0,scale_noise=5,l1=1,l2=1,l3=1), True

# data, plot_dynamics = generate_Sarcos(n_train, None, in_dim_niw, out_dim, seed, all=True), True
# data, plot_dynamics = generate_Barret(n_train, None, in_dim_niw, out_dim, seed, all=True), True
#
# data, data_test = normalize_data(data, 1),normalize_data(data_test,1)


# define gating
if stick_breaking == False:
    gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, ))*100)
    gating_prior = distributions.Dirichlet(**gating_hypparams)
else:
    gating_hypparams = dict(K=nb_models, gammas=np.ones((nb_models, )), deltas=np.ones((nb_models, )))
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

        # components_hypparams = dict(mu=np.zeros((in_dim_niw,)),
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
    # print('models_and_scores[0][1]',models_and_scores[0][1])

    # print labels
    for l in dpglm.labels_list:
        label = l.z
    # print(label)
    print('used label',dpglm.used_labels)
    print('# used labels',len(dpglm.used_labels))

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
            endeffector_pos(data, in_dim_niw, pred_y, 'results/kin_train.pdf')
        if plot_dynamics == True:
            # plot of inverse dynamics of first joint: motor torque and predicted motor torque
            motor_torque(n_train, data, pred_y, in_dim_niw)



    start_prediction = timeit.default_timer()

    if stick_breaking == False:
        alphas = 0
        alpha_hat = 0
        # for idx, g in enumerate(dpglm.gating):
        alphas = dpglm.gating.posterior.alphas
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                alpha_hat = alpha_hat + alphas[idx]

    mean_function = np.empty_like(data_test[:,in_dim_niw:])
    plus_std_function = np.empty_like(data_test[:,in_dim_niw:])
    minus_std_function = np.empty_like(data_test[:,in_dim_niw:])
    marg = np.zeros([len(dpglm.components)])
    stud_t = np.zeros([len(dpglm.components)])
    err = 0
    err_squared = 0
    D = in_dim_niw

    dot_xx = np.zeros([len(dpglm.components)])
    dot_yx = np.zeros([len(dpglm.components)])
    dot_yy = np.zeros([len(dpglm.components)])
    # for idx, c in enumerate(dpglm.components):
    #     if idx in dpglm.used_labels:
    #         data_cluster = [l.data[l.z == idx] for l in dpglm.labels_list]
    #         print('data_cluster',data_cluster)
    #         print('data_cluster[0]', data_cluster[0])
    #         x = data_cluster[0][:,:-out_dim]
    #         y = data_cluster[0][:,in_dim_niw:]
    #         dot_xx[idx] = np.dot(x, x.T)
    #         print('x',x)
    #         print('y',y)
    #         print(x.shape, y.shape)
    #         print(np.dot(y, x.T))
    #         dot_yx[idx] = np.dot(y, x.T)
    #         dot_yy[idx] = np.dot(y, y.T)

    for i in range(len(data_test[:,:-out_dim])):
        x_hat = data_test[i,:-out_dim]
        y_hat = data_test[i,in_dim_niw:]

        alphas_marg_sum = 0
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw = get_component_standard_parameters(c.posterior)
                marg[idx] = NIW_marg_likelihood(x_hat, mu, kappa, psi_niw, nu_niw, 1, 1)
                stud_t[idx] = student_t(x_hat, mu, kappa, psi_niw, nu_niw, D)
                # alphas_marg_sum = alphas_marg_sum + alphas[idx] * marg[idx]
                alphas_marg_sum = alphas_marg_sum + alphas[idx] * marg[idx]
        term = 0
        term_plus_std = 0
        term_minus_std = 0
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw = get_component_standard_parameters(c.posterior)
                S_0, N_0 = get_component_standard_parameters(c.prior)[6], get_component_standard_parameters(c.prior)[7]

                mat_T, std_T = matrix_t(data,idx,label,out_dim,in_dim_niw,affine,x_hat, V, M, nb_models, S_0, dot_xx[idx], dot_yx[idx], dot_yy[idx], psi_mniw)
                # stud_t = student_t(x_hat, mu, kappa, psi_niw, nu_niw, D)
                # print(mat_T, std_T)
                # term = term + mat_T * marg[idx] * alphas[idx] / alphas_marg_sum
                term = term + mat_T * marg[idx] / marg.sum()
                # term = term + mat_T #* mu #* alphas[idx]
                # term = term + mat_T * alphas[idx] * marg[idx]  / alphas_marg_sum * len(dpglm.components)
                # term = term + mat_T * stud_t[idx] / stud_t.sum() * alphas[idx] * len(dpglm.used_labels)
                # term = term + mat_T * stud_t[idx] / marg[idx] * alphas[idx]

                # if i == 1:
                #     print('stud_t[idx]',stud_t[idx])
                #     print('stud_t.sum()',stud_t.sum())
                #     print('stud_t[idx] / stud_t.sum()',stud_t[idx] / stud_t.sum())
                #     print('mu',mu)
                #     print('x_hat',x_hat)
                #     print('-------------')
                # 90% confidence interval
                # term_plus_std = term_plus_std + (mat_T + 0.1645 * std_T) * marg[idx] / marg.sum()
                # term_minus_std = term_minus_std + (mat_T - 0.1645 * std_T) * marg[idx] / marg.sum()
                term_plus_std = term_plus_std + (mat_T + 0.2 * std_T) * marg[idx] * alphas[idx] / alphas_marg_sum
                term_minus_std = term_minus_std + (mat_T - 0.2 * std_T) * marg[idx] * alphas[idx] / alphas_marg_sum

                #/ np.sqrt(alphas[idx])

        if out_dim > 1:
            term = term[:, 0]
        mean_function[i,:] = term #/ alpha_hat

        plus_std_function[i,:] = term_plus_std
        minus_std_function[i,:] = term_minus_std

        err = err + np.absolute((y_hat - mean_function[i]))
        err_squared = err_squared + ((y_hat - mean_function[i]) ** 2)

    if plot_prediction:
        if in_dim_niw + out_dim == 2:
            # plot_prediction_2d(data_test, mean_function)
            plot_prediction_2d_mean(data_test, mean_function, plus_std_function, minus_std_function)

        # plot of kinematics data
        if plot_kinematics == True:
            # plot of prediction for endeffector positions vs. data
            endeffector_pos(data_test, in_dim_niw, mean_function, 'results/kin_test.pdf')
        if plot_dynamics == True:
            # plot of inverse dynamics of first joint: motor torque and predicted motor torque
            motor_torque(n_train, data_test, mean_function, in_dim_niw)

    nMSE = calc_error_metrics(data_test, n_test, in_dim_niw, err, err_squared)
    # write nMSE for each iteration
    f = open('nMSE.txt', 'a+')
    f.write(str(nMSE[0]))
    f.write("\n")
    f.close()



# timer
stop = timeit.default_timer()
print('Plotting time: ', start_prediction - start_plotting)
print('Prediction time: ', stop - start_prediction)
print('Overall time: ', stop - start)


