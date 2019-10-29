import copy

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

from mimo import distributions, models
from mimo.util.text import progprint_xrange

import operator


n_samples = 2000

data = np.zeros((n_samples, 2))
step = 4. * np.pi / n_samples

for i in range(data.shape[0]):
    x = i * step - 6.
    data[i, 0] = x + npr.normal(0, 0.1)
    data[i, 1] = 3. * (np.sin(x) + npr.normal(0, .1))

plt.figure()
plt.plot(data[:, 0], data[:, 1], 'kx')
plt.title('data')


nb_models = 5


gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, )))
gating_prior = distributions.Dirichlet(**gating_hypparams)


in_dim = 1
out_dim = 1
affine = True
if affine:
    in_dim = in_dim + 1

components_hypparams = dict(M=np.zeros((out_dim, in_dim)),
                 V=1. * np.eye(in_dim),
                 # V= np.asarray([[1, 0],[0, 25]]),
                 affine=affine,
                 psi=np.eye(out_dim)*0.01,
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
    scores = []
    for _ in progprint_xrange(100):

        gmm.resample_model()
        for idx, l in enumerate(gmm.labels_list):
            l.r = l.get_responsibility()

        scores.append(gmm._vlb())

    allscores.append(scores)
    allmodels.append(copy.deepcopy(gmm))

plt.figure()
for scores in allscores:
    plt.plot(scores)
plt.title('model vlb scores vs iteration')

models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)],
                           key=operator.itemgetter(1), reverse=True)

gmm = models_and_scores[0][0]

# print labels
for l in gmm.labels_list:
    label = l.z
    print(label)
print('used labels',gmm.used_labels)

# prediction
componentA = np.zeros([len(gmm.components),np.size(gmm.components[0].A)])
for idx, component in enumerate(gmm.components):
    componentA[idx] = component.A

if affine:
    in_dim_pred = in_dim - 1
else:
    in_dim_pred = in_dim
pred_y = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    idx = gmm.labels_list[0].z[i]
    if affine:
        pred_y[i] = np.matmul(componentA[idx,:-1], data[i,:-out_dim].T) + componentA[idx,in_dim_pred:] * 1
    else:
        pred_y[i] = np.matmul(componentA[idx,:], data[i,:-out_dim].T)

plt.figure()
plt.plot(data[:, 0], data[:, 1], 'kx')
plt.scatter(data[:,0], pred_y, c='red', s=2)
plt.title('best model')
plt.show()