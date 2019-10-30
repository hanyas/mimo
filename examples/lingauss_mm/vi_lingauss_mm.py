import copy

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

from mimo import distributions, models
from mimo.util.text import progprint_xrange

import operator


n_samples = 2000

data = np.zeros((n_samples, 2))
step = 14. * np.pi / n_samples

for i in range(data.shape[0]):
    x = i * step - 6.
    data[i, 0] = x + npr.normal(0, 0.1)
    # data[i, 0] = x + npr.normal(0, 10)
    data[i, 1] = 3. * (np.sin(x) + npr.normal(0, .1))

# # Normalize data to 0 mean, 1 std_deviation
# scaling = 1.0
# mean = np.mean(data, axis=0)
# std_deviation = np.std(data,axis=0)
# data = ( data - mean ) / ( std_deviation * scaling)

plt.figure()
plt.plot(data[:, 0], data[:, 1], 'kx')
plt.title('data')

nb_models = 25




gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, ))*1)
gating_prior = distributions.Dirichlet(**gating_hypparams)

affine = True
out_dim = 1
if affine:
    in_dim = data.shape[1] - out_dim + 1
else:
    in_dim = data.shape[1] - out_dim
print(in_dim)

components_hypparams = dict(M=np.zeros((out_dim, in_dim)),
                 # V=1 * np.eye(in_dim),
                 V= np.asarray([[1, 0],[0, 100]]),
                 affine=affine,
                 psi=np.eye(out_dim)*0.001,
                 nu=2 * out_dim + 1)
components_prior = distributions.MatrixNormalInverseWishart(**components_hypparams)

gmm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                     components=[distributions.BayesianLinearGaussian(components_prior) for _ in range(nb_models)])

gmm.add_data(data)

allscores = []
allmodels = []
for superitr in range(1):
    # Gibbs sampling to wander around the posterior
    print('Gibbs Sampling')
    for _ in progprint_xrange(50):
        gmm.resample_model()

    # mean field to lock onto a mode
    print('Mean Field')
    scores = [gmm.meanfield_coordinate_descent_step() for _ in progprint_xrange(200)]

    allscores.append(scores)
    allmodels.append(copy.deepcopy(gmm))

plt.figure()
for scores in allscores:
    plt.plot(scores)
plt.title('model vlb scores vs iteration')

models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)],
                           key=operator.itemgetter(1), reverse=True)


plt.show()

gmm = models_and_scores[0][0]

print('used label',gmm.used_labels)
componentA = np.zeros([len(gmm.components),np.size(gmm.components[0].A)])
for idx, component in enumerate(gmm.components):
    componentA[idx] = component.A

print(componentA)

for l in gmm.labels_list:
    label = l.z
    print(label)

pred_y = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    idx = gmm.labels_list[0].z[i]
    if affine:
        pred_y[i] = componentA[idx,0] * data[i, 0] + componentA[idx,1] * 1
    else:
        pred_y[i] = componentA[idx,0] * data[i, 0]

plt.figure()
plt.plot(data[:, 0], data[:, 1], 'kx')
plt.scatter(data[:,0], pred_y, c='red', s=2)
plt.title('best model')
plt.show()

