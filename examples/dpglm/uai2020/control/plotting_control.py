import os

import mimo

import numpy as np
import matplotlib.pyplot as plt


# set working directory
# evalpath = os.path.abspath(mimo.__file__ + '/../../examples/dpglm/uai2020/control')
evalpath = os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020/control')

os.chdir(evalpath)

# set parameters for plot
y_axis = 'used-models'             # nmse, used-models, nlpd, evar, mse

datasets = ['cartpole']        # pendulum, cartpole
priors = ['stick-breaking', 'dirichlet']   # dirichlet, stick-breaking
x_axises = ['alpha']   # alpha, models, horizon


# iterate all 4 plots:
for n in range(len(x_axises)):
    for l in range(len(datasets)):

        # create a figure for each choice of x_axis and dataset
        fig, ax2 = plt.subplots()
        dataset = datasets[l]
        plt.title(dataset)

        # two priors in one plot (dirichlet / stick-breaking)
        for m in range(len(priors)):
            prior = priors[m]
            x_axis = x_axises[n]

            # set x-ticks
            alpha = []
            if prior == 'dirichlet':
                alpha = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
            if prior == 'stick-breaking':
                alpha = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

            models = [50, 75, 100, 125, 150]

            horizon = [1, 5, 10, 15, 20, 25]

            # get data from saved files
            iterator = []
            if x_axis == 'models':
                iterator = models
            elif x_axis == 'alpha':
                iterator = alpha
            elif x_axis == 'horizon':
                iterator = horizon
            metrics = np.zeros((len(iterator), 12))

            for i in range(len(iterator)):
                path = os.path.join(str(evalpath), str(dataset), str(dataset) + '_' + str(x_axis),
                                    str(dataset) + '_' + str(prior) + '_' + str(x_axis) + '_' + str(iterator[i]) + '.csv')

                with open(path) as mycsv:
                    count = 0
                    for line in mycsv:
                        metrics[i, count] = line
                        count += 1
                        if count == 12:
                            break

            # column indices for what to show on y-axis:
            # 0, 1 = mean_mse, std_mse
            # 2, 3 = mean_nmse, std_nmse
            # 4, 5 = mean_evar, std_evar
            # 6, 7 = mean_nb_models, std_nb_models
            # 8, 9 = mean_duration, std_duration
            # 10, 11 = mean_nlpd, std_nlpd (negative log predictive density)

            # plot nmse or used models on y-axis
            x, y = [], []
            error = []
            if y_axis == 'mse':
                y = metrics[:, 0]       # choose mean_nmse for y-axis
                error = metrics[:, 1]   # choose std_nmse as error
                ax2.set_ylabel('mse')

            if y_axis == 'nmse':
                y = metrics[:, 2]       # choose mean_nmse for y-axis
                error = metrics[:, 3]   # choose std_nmse as error
                ax2.set_ylabel('nmse')

            if y_axis == 'evar':
                y = metrics[:, 4]       # choose mean_nmse for y-axis
                error = metrics[:, 5]   # choose std_nmse as error
                ax2.set_ylabel('evar')

            if y_axis == 'used-models':
                y = metrics[:, 6]       # choose mean_nb_models for y-axis
                error = metrics[:, 7]   # choose std_nb_models as error
                ax2.set_ylabel('used models')

            if y_axis == 'nlpd':
                y = metrics[:, 10]       # choose mean_nmse for y-axis
                error = metrics[:, 11]   # choose std_nmse as error
                ax2.set_ylabel('nlpd')

            # set the x-axis of the stick-breaking prior
            if m == 0:
                ax2.xaxis.label.set_color('red')
                ax2.tick_params(axis='x', colors='red')

                if x_axis == 'alpha':
                    x = alpha
                    plt.xscale('log')  # log scale on alphas
                    ax2.set_xlabel('log(alpha) - stick-breaking prior')

                if x_axis == 'models':
                    x = models
                    ax2.set_xlabel('max. number of models - stick-breaking prior')

                if x_axis == 'horizon':
                    x = horizon
                    ax2.set_xlabel('prediction horizon - stick-breaking prior')

                ax2.errorbar(x, y, yerr=error, fmt='-o', capsize=7, c='red', markersize=5)  # fillstyle='none'

            # set the x-axis of the dirichlet prior
            if m == 1:
                ax1 = ax2.twiny()
                ax1.xaxis.label.set_color('blue')
                ax1.tick_params(axis='x', colors='blue')

                if x_axis == 'alpha':
                    x = alpha
                    plt.xscale('log')  # log scale on alphas
                    ax1.set_xlabel('log(alpha) - dirichlet prior')

                if x_axis == 'models':
                    x = models
                    ax1.set_xlabel('max. number of models - dirichlet prior')

                if x_axis == 'horizon':
                    x = horizon
                    ax1.set_xlabel('prediction horizon - dirichlet prior')

                ax1.errorbar(x, y, yerr=error, fmt='-x', capsize=7)

        # save tikz and pdf
        import tikzplotlib
        # path = os.path.join(str(dataset) + '/' + dataset + '_' + x_axis + '/')
        path = os.path.join(str(dataset) + '/')

        plt.tight_layout()  # otherwise title is clipped off

        # print(path, dataset, x_axis, y_axis)
        # tikzplotlib.save(path + dataset + '_' + x_axis + '_' + y_axis + '.tex')
        # plt.savefig(path + dataset + '_' + x_axis + '_' + y_axis + '.pdf')

        plt.show()
