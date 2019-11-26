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
plt.show()

nb_models = 5

gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models, )))
gating_prior = distributions.Dirichlet(**gating_hypparams)

affine = True
out_dim = 1
in_dim = 1

if affine:
    n_params = in_dim + 1
else:
    n_params = in_dim

components_hypparams = dict(M=np.zeros((out_dim, n_params)),
                            V=1. * np.eye(out_dim),
                            affine=affine,
                            psi=np.eye(out_dim) * 0.01,
                            nu=2 * out_dim + 1)
components_prior = distributions.MatrixNormalInverseWishart(**components_hypparams)

model = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                       components=[distributions.BayesianLinearGaussian(components_prior) for _ in range(nb_models)])

model.add_data(data)

allscores = []
allmodels = []
for superitr in range(5):
    # Gibbs sampling to wander around the posterior
    print('Gibbs Sampling')
    scores = []
    for _ in progprint_xrange(300):
        model.resample_model()
        for idx, l in enumerate(model.labels_list):
            l.r = l.get_responsibility()
        scores.append(model._vlb())  # A hack to compare models

    allscores.append(scores)
    allmodels.append(copy.deepcopy(model))

plt.figure()
for scores in allscores:
    plt.plot(scores)
plt.title('model vlb scores vs iteration')

models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)],
                           key=operator.itemgetter(1), reverse=True)
model = models_and_scores[0][0]

# print labels
for l in model.labels_list:
    label = l.z
    print(label)
print('used labels', model.used_labels)

# prediction
componentA = np.zeros([len(model.components), np.size(model.components[0].A)])
for idx, component in enumerate(model.components):
    componentA[idx] = component.A

pred_y = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    idx = model.labels_list[0].z[i]
    if affine:
        pred_y[i] = np.matmul(componentA[idx,:-1], data[i, :-out_dim].T) + componentA[idx, in_dim:] * 1
    else:
        pred_y[i] = np.matmul(componentA[idx,:], data[i, :-out_dim].T)

plt.figure()
plt.plot(data[:, 0], data[:, 1], 'kx')
plt.scatter(data[:, 0], pred_y, c='red', s=2)
plt.title('best model')
plt.show()
