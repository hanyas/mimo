import os
os.environ["OMP_NUM_THREADS"] = "1"

import mimo

import numpy as np
import matplotlib.pyplot as plt


# set working directory
evalpath = os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020/')
os.chdir(evalpath)



# set parameters for plot
y_axis = 'used-models'      # smse, used-models

dataset = 'pendulum'        # pendulum, cartpole
prior  = 'stick-breaking'   # dirichlet, stick-breaking
x_axis  = 'models'          # alpha, models


# dataset_choices = ['pendulum', 'cartpole']        # pendulum, cartpole
# prior_choices = ['stick-breaking', 'dirichlet']   # dirichlet, stick-breaking
# x_axis_choices = ['models', 'alpha']              # alpha, models
#
# # automatically iterate all 8 plots:
# for n in range(len(x_axis_choices)):
#     for l in range(len(dataset_choices)):
#         for m in range(len(prior_choices)):
#
#             dataset = dataset_choices[l]
#             prior = prior_choices[m]
#             x_axis = x_axis_choices[n]

if prior == 'dirichlet':
    alpha = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
if prior == 'stick-breaking':
    alpha = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

if dataset == 'pendulum':
    models = [30, 45, 60, 75, 90]
if dataset == 'cartpole':
    models = [50, 75, 100, 125, 150]

if x_axis == 'models':
    iterator = models
if x_axis == 'alpha':
    iterator = alpha

# get data from saved files
metrics = np.zeros((len(iterator), 10))
for i in range(len(iterator)):
    path = os.path.join(evalpath + '\\' + dataset + '\\' + dataset + '_' + x_axis + '\\' +
dataset + '_' + x_axis + '_' + prior + '_' + str(iterator[i]) + '.csv')
    with open(path) as mycsv:
        count = 0
        for line in mycsv:
            metrics[i, count] = line
            count += 1

plt.figure()

# column indices for what to show on y-axis:
# 0, 1 = mean_mse, std_mse
# 2, 3 = mean_smse, std_smse
# 4. 5 = mean_evar, std_evar
# 6, 7 = mean_nb_models, std_nb_models
# 8, 9 = mean_duration, std_duration

# plot alphas or models on x-axis
# plot nmse or used models on y-axis
if x_axis == 'alpha':
    x = alpha
    plt.xscale('log')       # log scale on alphas
    plt.xlabel('alphas')

if x_axis == 'models':
    x = models
    plt.xlabel('max. number of models')

if y_axis == 'used-models':
    y = metrics[:, 6]       # choose mean_nb_models for y-axis
    error = metrics[:, 7]   # choose std_nb_models as error
    plt.ylabel('used models')

if y_axis == 'smse':
    y = metrics[:, 2]       # choose mean_smse for y-axis
    error = metrics[:, 3]   # choose std_smse as error
    plt.ylabel('smse')

plt.title(dataset + ' dataset - ' + prior + ' prior ')

plt.errorbar(x, y, yerr=error, fmt='-o', capsize=10)

# set working directory
evalpath = os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020')
os.chdir(evalpath)

# save tikz and pdf
import tikzplotlib
# path = os.path.join(str(dataset) + '/' + dataset + '_' + x_axis + '/')
path = os.path.join(str(dataset) + '/')
tikzplotlib.save(path + dataset + '_' + x_axis +  '_' + prior + '_' + y_axis + '.tex')
plt.savefig(path + dataset + '_' + x_axis + '_' + prior +  '_' + y_axis + '.pdf')

plt.show()
