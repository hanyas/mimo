import copy

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mimo import distributions, models
from mimo.util.text import progprint_xrange

import operator
import random
import timeit

# set random seeed
np.random.seed(seed=26)
random.seed(26)

# start timer
start = timeit.default_timer()

# set dimensions
affine = True

in_dim_niw = 1
out_dim = 1
if affine:
    in_dim_mniw = in_dim_niw + 1
else:
    in_dim_mniw = in_dim_niw

# # load Cosmic Microwave Background (CMB) training_data from Hannah (2011)
# data = np.genfromtxt("cmb.csv", dtype=None, encoding=None, usecols=(0, 1))
# np.random.shuffle(data)
#
# # generate subset of training_data points
# data = data[0:400,:]
# # n_samples = 500
# # training_data = training_data[np.random.choice(training_data.shape[0], size=n_samples, replace=False), :]
#
# # n_training = 400
# # training_data = data[:n_training, :]
# # test_data = data[n_training:, :]

# create sin data
n_samples = 2000
nb_models = 30 # best mit 30 modelle

data = np.zeros((n_samples, in_dim_niw + out_dim))
step = 14. * np.pi / n_samples

for i in range(data.shape[0]):
    x = i * step - 6.
    data[i, 0] = (x + npr.normal(0, 0.1))
    data[i, 1] = (3. * (np.sin(x) + npr.normal(0, .1)))
    # data[i, 1] = (12. * (np.sin(x) + npr.normal(0, .1)))
# np.random.shuffle(data)

# # Normalize data to 0 mean, 1 std_deviation
# scaling = 1.0
# mean = np.mean(data, axis=0)
# std_deviation = np.std(data,axis=0)
# data = (data - mean) / (std_deviation * scaling)

# # Center data to 0 mean
# scaling = 1
# mean = np.mean(data, axis=0)
# data = (data - mean) / scaling

# define gating
gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, ))*1)
gating_prior = distributions.Dirichlet(**gating_hypparams)
# gating_hypparams = dict(K=nb_models, gammas=np.ones((nb_models, )), deltas=np.ones((nb_models, ))*1)
# gating_prior = distributions.StickBreaking(**gating_hypparams)

# # define components
# components_hypparams = dict(mu=np.zeros((in_dim_niw,)),
# # components_hypparams = dict(mu=npr.uniform(-10,40,size=in_dim_niw),
#                  kappa=0.05, #0.05
#                  psi_niw=np.eye(in_dim_niw),
#                  nu_niw=2 * in_dim_niw + 1,
#                  M=np.zeros((out_dim, in_dim_mniw)),
#                  # V=10. * np.eye(in_dim_mniw),
#                  V= np.asarray([[1, 0],[0, 30]]),    #Fixme: 1,0,0,10
#                  affine=affine,
#                  psi_mniw=np.eye(out_dim) * 0.01,   #Fixme: 0.01 (eher) oder 0.001??
#                  nu_mniw=2 * out_dim + 1)
# components_prior = distributions.NormalInverseWishartMatrixNormalInverseWishart(**components_hypparams)


# define components
components_prior = []
for j in range(nb_models):

    rnd_V1 = npr.uniform(0.1,10)            #Fixme best: 0.1,10
    rnd_V2 = npr.uniform(1, 1000)             #Fixme best: 1,1000
    rnd_psi_mniw = npr.uniform(0.001, 0.1)   #Fixme: best: 0.001, 0.1
    # rnd_V1 = np.absolute(npr.normal(1, 10))
    # rnd_V2 = np.absolute(npr.normal(10, 100))
    # rnd_psi_mniw = npr.normal(0.001, 1)

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
                                # V=10. * np.eye(in_dim_mniw),
                                V= np.asarray([[rnd_V1, 0], [0, rnd_V2]]),  #Fixme: 1,0,0,10
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
for superitr in range(5):

    # # Gibbs sampling to wander around the posterior
    print('Gibbs Sampling')
    scores = []
    for _ in progprint_xrange(250):
        gmm.resample_model()

        # for idx, l in enumerate(gmm.labels_list):
        #     l.r = l.get_responsibility()
        # scores.append(gmm._vlb())

    # print('Stochastic Mean Field')
    # minibatchsize = 30
    # prob = minibatchsize / float(n_samples)
    # for _ in progprint_xrange(300):
    #     minibatch = npr.permutation(n_samples)[:minibatchsize]
    #     gmm.meanfield_sgdstep(minibatch=data[minibatch], prob=prob, stepsize=1e-3)
    #
    # allscores.append(gmm.meanfield_coordinate_descent_step())
    # allmodels.append(copy.deepcopy(gmm))

    # mean field to lock onto a mode
    # gmm.resample_model()
    print('Mean Field')
    scores = [gmm.meanfield_coordinate_descent_step() for _ in progprint_xrange(75)]

    allscores.append(scores)
    allmodels.append(copy.deepcopy(gmm))

# plot scores
plt.figure()
for scores in allscores:
    plt.plot(scores)
plt.title('model vlb scores vs iteration')
plt.show()

# Sort models
# VI
models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)],
                           key=operator.itemgetter(1), reverse=True)
# SVI
# models_and_scores = sorted([(m, s) for m, s in zip(allmodels, allscores)],
#                            key=operator.itemgetter(1), reverse=True)
gmm = models_and_scores[0][0]
print('models_and_scores[0][1]',models_and_scores[0][1])

# print labels
for l in gmm.labels_list:
    label = l.z
    print(label)
print('used label',gmm.used_labels)
print('# used labels',len(gmm.used_labels))

# prediction
componentA = np.zeros([len(gmm.components),np.size(gmm.components[0].A)])
component_mu = np.zeros([len(gmm.components),np.size(gmm.components[0].mu)])
for idx, component in enumerate(gmm.components):
    componentA[idx] = component.A
    component_mu[idx] = component.mu

pred_y = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    idx = gmm.labels_list[0].z[i]
    if affine:
        pred_y[i] = np.matmul(componentA[idx,:-1], data[i,:-out_dim].T) + componentA[idx,in_dim_niw:] * 1
    else:
        pred_y[i] = np.matmul(componentA[idx,:], data[i,:-out_dim].T)

# 2-dimensional plot of prediction
plt.scatter(data[:, 0], data[:, 1], s=1, zorder=1)
plt.scatter(data[:, 0], pred_y, c='red', s=1, zorder=2)
plt.title('best model')
plt.show()

# 3-dimensional plot of prediction
# fig = plt.figure()
# ax = Axes3D(fig)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(data[:, 0], data[:, 1], data[:, 2], 'kx', zorder=1)
# ax.scatter(data[:, 0], data[:, 1], pred_y, c='red', s=3, zorder=2)
# # ax.title('best model')
# plt.show()

# timer
stop = timeit.default_timer()
print('Overall time: ', stop - start)
