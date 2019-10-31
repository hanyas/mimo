import copy

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# import py2LinkArm

from mimo import distributions, models
from mimo.util.text import progprint_xrange
from mimo.util.data import generate_CMB, generate_SIN, generate_kinematics, generate_heaviside1, generate_heaviside2, generate_gaussian

import operator
import random
import timeit

# start timer
start = timeit.default_timer()

# set random seed
seed = None
np.random.seed(seed=seed)
random.seed(seed)


# settings
affine = True
nb_models = 25
superitr = 1

# set inference methods
gibbs = True
gibbs_iter = 250

mf = True       #mf with fixed iterations
mf_conv = False  #mf with convergence criterion
mf_sgd = False

# generate data
in_dim_niw = 1
out_dim = 1
n_train = 2000

# data needs to be of shape in_dim_niw = 1, out_dim = 1
data = generate_SIN(n_train,in_dim_niw, out_dim, freq=14, shuffle=False, seed=seed)
# data = generate_heaviside2(n_train,scaling=1)
# data = generate_CMB(n_train, seed)

# data needs to be of shape in_dim_niw = 1, 2 or 3 and out_dim = 1 or 2
# data = generate_kinematics(n_train=n_train,out_dim=out_dim,num_joints=in_dim_niw,loc_noise=0,scale_noise=5,l1=1,l2=1,l3=1)

# data can be of arbitrary dimensions
# data = generate_gaussian(n_train, out_dim, in_dim_niw)



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

    # rnd_V1 = npr.uniform(0.1,10)            #Fixme best: 0.1,10
    # rnd_V2 = npr.uniform(1, 1000)             #Fixme best: 1,1000
    V = np.eye(in_dim_mniw)
    for i in range(in_dim_mniw):
        if i < in_dim_mniw - 1:
            V[i, i] = npr.uniform(0.1, 10)
        else:
            V[i, i] = npr.uniform(1, 1000) # special treatment for offset
    rnd_psi_mniw = npr.uniform(0.001, 0.1)   #Fixme: best: 0.001, 0.1

    mu_low = np.amin(data[:,:-out_dim])
    mu_high = np.amax(data[:,:-out_dim])

    rnd_psi_niw = npr.uniform(0.1, 10)      #Fixme: const
    rnd_kappa = npr.uniform(0.01,0.1)        #Fixme: const



    # components_hypparams = dict(mu=np.zeros((in_dim_niw,)),
    components_hypparams = dict(mu=npr.uniform(mu_low,mu_high,size=in_dim_niw),
                                kappa= rnd_kappa,  #0.05
                                psi_niw=np.eye(in_dim_niw) * rnd_psi_niw,
                                nu_niw=2 * in_dim_niw + 1,
                                M=np.zeros((out_dim, in_dim_mniw)),  #+np.ones((out_dim, in_dim_mniw))*rnd_psi_miw,
                                V=V,
                                # V= np.asarray([[rnd_V1, 0], [0, rnd_V2]]),  #Fixme: 1,0,0,10
                                affine=affine,
                                psi_mniw=np.eye(out_dim) * rnd_psi_mniw,  #Fixme: 0.01 (eher) oder 0.001??
                                nu_mniw=2 * out_dim + 1)
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
for _ in range(superitr):
    scores = []

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
        scores = [gmm.meanfield_coordinate_descent_step() for _ in progprint_xrange(250)]

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

    allscores.append(scores)
    allmodels.append(copy.deepcopy(gmm))

print(allscores)
# plot scores
plt.figure()
for scores in allscores:
    plt.plot(scores)
plt.title('model vlb scores vs iteration')
plt.show()

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

pred_y = np.zeros((data.shape[0], out_dim))
err = 0
for i in range(data.shape[0]):
    idx = gmm.labels_list[0].z[i]
    x = data[i,:-out_dim]
    y = data[i,in_dim_niw:]
    pred_y[i] = gmm.components[idx].predict(x)
    err = err + ((y - pred_y[i])**2)
    # print(err)
# err = err[0] + err[1]
mse = np.mean(err)
print(mse)

# 2-dim plot of prediction
if in_dim_niw + out_dim == 2:
    plotting = True # switch on plotting (only 2d data)
else:
    plotting = False
if plotting:
    plt.scatter(data[:, 0], data[:, 1], s=1, zorder=1)
    plt.scatter(data[:, 0], pred_y, c='red', s=1, zorder=2)
    plt.title('best model')
    plt.show()


# timer
stop = timeit.default_timer()
print('Overall time: ', stop - start)
