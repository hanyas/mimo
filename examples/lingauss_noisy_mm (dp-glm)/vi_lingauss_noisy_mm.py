import numpy as np
import numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D

from mimo import distributions, models
from mimo.util.text import progprint_xrange
from mimo.util.data import generate_CMB, generate_SIN, generate_kinematics, generate_heaviside1, generate_heaviside2, generate_gaussian, generate_Sarcos, generate_Barret,generate_kinematics_cos
from mimo.util.plot import plot_absolute_error, plot_scores, plot_nMSE, plot_prediction_2d, endeffector_pos, motor_torque
from mimo.util.error_metrics import calc_error_metrics
from mimo.util.prediction import predict_train

import copy
import operator
import random
import timeit

# misc
# timer
start = timeit.default_timer()
# set random seed
seed = None
np.random.seed(seed=seed)
random.seed(seed)
# set booleans to standard values
plot_kinematics = False
plot_dynamics = False



# settings
affine = True
nb_models = 25
superitr = 1

# set inference methods
gibbs = True
gibbs_iter = 250

mf = False  #mf with fixed iterations
mf_iter = 100

mf_conv = True #mf with convergence criterion
mf_sgd = False

# generate data
in_dim_niw = 1
out_dim = 2
n_train = 3000

# data needs to be of shape in_dim_niw = 1, out_dim = 1
# data = generate_SIN(n_train,in_dim_niw, out_dim, freq=14, shuffle=False, seed=seed)
# data = generate_heaviside2(n_train,scaling=1)
# data = generate_CMB(n_train, seed)

# data can be of arbitrary dimensions
# data = generate_gaussian(n_train, out_dim, in_dim_niw)

# kinematics - data shape in_dim_niw = 1, 2 or 3 and out_dim = 2
data, plot_kinematics = generate_kinematics(n_train=n_train,out_dim=out_dim,num_joints=in_dim_niw,loc_noise=0,scale_noise=5,l1=1,l2=1,l3=1), True

# inverse dynamics - data shape in_dim_niw = 3 and out_dim = 1
# data, plot_dynamics = generate_Sarcos(n_train, None, in_dim_niw, out_dim, seed, all=True), True
# data, plot_dynamics = generate_Barret(n_train, None, in_dim_niw, out_dim, seed, all=True), True






# define gating
gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, ))*1)
gating_prior = distributions.Dirichlet(**gating_hypparams)

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
            V[i, i] = npr.uniform(0.1, 10)      #Fixme best: 0.1,10
        else:
            V[i, i] = npr.uniform(1, 1000) # offset, Fixme best: 1,1000
    # rnd_psi_mniw = npr.uniform(0.001, 0.1)   #Fixme: best: 0.001, 0.1
    nu_mniw = 2 * out_dim + 1

    mu_low = np.amin(data[:,:-out_dim])
    mu_high = np.amax(data[:,:-out_dim])

    rnd_psi_niw = npr.uniform(0.1, 10)      #Fixme: const, 0.1, 10
    rnd_kappa = npr.uniform(0.01,0.1)        #Fixme: const, 0.01, 0.1

    # components_hypparams = dict(mu=np.zeros((in_dim_niw,)),
    components_hypparams = dict(mu=npr.uniform(mu_low,mu_high,size=in_dim_niw),
                                kappa= rnd_kappa,  #0.05
                                psi_niw=np.eye(in_dim_niw) * rnd_psi_niw,
                                nu_niw=2 * in_dim_niw + 1,
                                M=np.zeros((out_dim, in_dim_mniw)),
                                V=V,
                                affine=affine,
                                psi_mniw=np.eye(out_dim) * nu_mniw,  #Fixme: 0.01 (eher) oder 0.001??
                                nu_mniw= nu_mniw )#2 * out_dim + 1)
    components_prior_rand = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)
    components_prior.append(components_prior_rand)

# define model
gmm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                     components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior[i]) for i in range(nb_models)])
# gmm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
#                      components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior) for _ in range(nb_models)])
# gmm = models.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
#                       components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior) for _ in range(nb_models)])

gmm.add_data(data)





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
            gmm.resample_model()
            if mf != True and mf_conv != True and mf_sgd != True:
            # this part is needed if Gibbs sampling is used without Mean Field
                for idx, l in enumerate(gmm.labels_list):
                    l.r = l.get_responsibility()
                scores.append(gmm._vlb())

    if mf == True:
        # mean field to lock onto a mode (without convergence criterion)
        print('Mean Field')
        if gibbs != True:
            gmm.resample_model()
        # scores = [gmm.meanfield_coordinate_descent_step() for _ in progprint_xrange(250)]
        for _ in progprint_xrange(mf_iter):
            scores.append(gmm.meanfield_coordinate_descent_step())

            pred_y, err, err_squared = predict_train(gmm, data, out_dim, in_dim_niw)
            var = np.var(data[:, in_dim_niw:], axis=0)
            MSE = 1 / n_train * err_squared
            nMSE.append(np.sum(MSE[0] / var[0]))
            err_.append(np.sum(err))

    if mf_conv == True:
        # mean field to lock onto a mode (with convergence criterion)
        print('Mean Field')
        if gibbs != True:
            gmm.resample_model()
        scores = gmm.meanfield_coordinate_descent(tol=1e-2, maxiter=500, progprint=True)

    # stochastic mean field to lock onto a mode
    if mf_sgd == True:
        print('Stochastic Mean Field')
        if gibbs != True:
            gmm.resample_model()
        minibatchsize = 30
        prob = minibatchsize / float(n_train)
        for _ in progprint_xrange(300):
            minibatch = npr.permutation(n_train)[:minibatchsize]
            gmm.meanfield_sgdstep(minibatch=data[minibatch], prob=prob, stepsize=1e-3)
            for idx, l in enumerate(gmm.labels_list):
                l.r = l.get_responsibility()
            scores.append(gmm._vlb())
        # allscores.append(gmm.meanfield_coordinate_descent_step())

    all_err.append(err_)
    all_nMSE.append(nMSE)
    allscores.append(scores)
    allmodels.append(copy.deepcopy(gmm))





start_plotting = timeit.default_timer()

# plot error metrics over iterations
plot_scores(allscores)
if mf == True:
    plot_nMSE(all_nMSE)
    # plot_absolute_error(all_err)

# Sort models and select best one
models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)],
                           key=operator.itemgetter(1), reverse=True)
gmm = models_and_scores[0][0]
print('models_and_scores[0][1]',models_and_scores[0][1])

# print labels
# for l in gmm.labels_list:
#     label = l.z
#     print(label)
print('used label',gmm.used_labels)
print('# used labels',len(gmm.used_labels))

# predict on training data
pred_y, err, err_squared = predict_train(gmm, data, out_dim, in_dim_niw)

# calculate and print error metrics (MAE, nMAE, MSE, nMSE, RMSE, nRMSE)
calc_error_metrics(data, n_train, in_dim_niw, err, err_squared)

# 2-dim plot of prediction
if in_dim_niw + out_dim == 2:
    plot_prediction_2d(data, pred_y)

# plot of kinematics data
if plot_kinematics == True:
    # plot of prediction for endeffector positions vs. data
    endeffector_pos(data, in_dim_niw, pred_y)
if plot_dynamics == True:
    # plot of inverse dynamics of first joint: motor torque and predicted motor torque
    motor_torque(n_train, data, pred_y, in_dim_niw)

# timer
stop_plotting = timeit.default_timer()
stop = timeit.default_timer()
print('Plotting time: ', stop_plotting - start_plotting)
print('Overall time: ', stop - start)
