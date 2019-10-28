import copy

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

from mimo import distributions, models
from mimo.util.text import progprint_xrange

import operator



n_samples = 500

data = np.zeros((n_samples, 3))
step = 4. * np.pi / n_samples

for i in range(data.shape[0]):
    x = i * step - 6.
    data[i, 0] = 1
    data[i, 1] = x + npr.normal(0, 0.1)
    data[i, 2] = 3. * (np.sin(x) + npr.normal(0, .1))
    # if i > 1000:
    #     data[i, 2] = data[i, 2] + 1


# step1 = 14. * np.pi / n_samples
# step2 = 7. * np.pi / n_samples
# for i in range(data.shape[0]):
#     x1 = i * step1 - 6.
#     x2 = i * step2 - 3.
#     data[i, 0] = x1 + npr.normal(0, 0.1)
#     data[i, 1] = x2 + npr.normal(0, 0.1)
#     data[i, 2] = 3. * (np.sin(x1) + np.cos(x2)  + npr.normal(0, .1))
#     data[i, 3] = 4. * (np.sin(x1) - np.cos(x2) + npr.normal(0, .1))



print('Data array:\n',data)
plt.figure()
plt.plot(data[:, 1], data[:, 2], 'kx')
plt.title('Data')
plt.show()

nb_models = 10


# gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, )))
# gating_prior = distributions.Dirichlet(**gating_hypparams)
gating_hypparams = dict(K=nb_models, gammas=np.ones((nb_models, )), deltas=np.ones((nb_models, )))
gating_prior = distributions.StickBreaking(**gating_hypparams)



# components_hypparams = dict(mu=np.mean(data, axis=0), kappa=0.01, psi=np.eye(2), nu=5)
# components_prior = distributions.NormalInverseWishart(**components_hypparams)



in_dim = 2
out_dim = 1
hypparams = dict(M=np.zeros((out_dim, in_dim)),
                 V=1. * np.eye(in_dim),
                 psi=np.eye(out_dim),
                 nu=2 * out_dim + 1)
components_prior = distributions.MatrixNormalInverseWishart(**hypparams)


# gmm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
#                      components=[distributions.BayesianLinearGaussian(components_prior) for _ in range(nb_models)])
gmm = models.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
                      components=[distributions.BayesianLinearGaussian(components_prior) for _ in range(nb_models)])

# in_dim = 2
# out_dim = 1
# hypparams = dict(mu=np.zeros((in_dim,)),
#                  kappa=0.05,
#                  psi_niw=np.eye(in_dim),
#                  nu_niw=2 * in_dim + 1,
#                  M=np.zeros((out_dim, in_dim)),
#                  V=1. * np.eye(in_dim),
#                  psi_mniw=np.eye(out_dim),
#                  # psi_mniw=np.array([1000]),
#                  nu_mniw=2 * out_dim + 1)
#
# components_prior = distributions.NormalInverseWishartMatrixNormalInverseWishart(**hypparams)
#
# # gmm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
# #                      components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior) for _ in range(nb_models)])
# gmm = models.Mixture(gating=distributions.BayesianCategoricalWithStickBreaking(gating_prior),
#                       components=[distributions.BayesianLinearGaussianWithNoisyInputs(components_prior) for _ in range(nb_models)])


gmm.add_data(data)


allscores = []
allmodels = []
for superitr in range(20):
    # Gibbs sampling to wander around the posterior
    print('Gibbs Sampling')
    for _ in progprint_xrange(100):
        gmm.resample_model()

    # mean field to lock onto a mode
    print('Mean Field')
    scores = [gmm.meanfield_coordinate_descent_step() for _ in progprint_xrange(100)]

    allscores.append(scores)
    allmodels.append(copy.deepcopy(gmm))

print(allscores)
plt.figure()
for scores in allscores:
    plt.plot(scores)
plt.title('model vlb scores vs iteration')

models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)],
                           key=operator.itemgetter(1), reverse=True)

# plt.figure()
# models_and_scores[0][0].plot()
# plt.title('best model')
plt.show()

print(gmm.components)

gmm = models_and_scores[0][0]

print('used label',gmm.used_labels)
componentA = np.zeros([len(gmm.components),np.size(gmm.components[0].A)])
for idx, component in enumerate(gmm.components):
    print('idx',idx)
    print('component.A',component.A)
    componentA[idx] = component.A
    # if idx == 0:
    #     componentA = component.A
    print('component.sigma',component.sigma)
#
#
# for idx in range(len(gmm.labels_list)):
#     print(gmm.labels_list[idx].z)
#     print(sum(gmm.labels_list[idx].z))
#     print(idx)
#     print(len(gmm.labels_list[idx].z))

for l in gmm.labels_list:
    label = l.z
    print(label)
# label_usages = sum(np.bincount(l.z, minlength=self.N) for l in self.labels_list)
# print(gmm.labels_list.z)
print('used label',gmm.used_labels)

# print(gmm.generate(len(gmm.components)))
print('A\n', componentA)

pred_y = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    idx = gmm.labels_list[0].z[i]
    # pred_y[i] = componentA[idx] * data[i,0]
    pred_y[i] = componentA[idx,0] * data[i, 0] + componentA[idx,1] * data[i,1]
plt.plot(data[:, 1], data[:, 2], 'kx')
plt.scatter(data[:,1], pred_y, c='red', s=2)
plt.show()



