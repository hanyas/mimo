import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import os

def plot_gaussian(mu, lmbda, color='rnd_psi_mniw', label='', alpha=1.0, ax=None,
                  artists=None):
    ax = ax if ax else plt.gca()

    t = np.hstack([np.arange(0, 2 * np.pi, 0.01), 0])
    circle = np.vstack([np.sin(t), np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda), circle)

    if artists is None:
        point = ax.scatter([mu[0]], [mu[1]], marker='D', color=color, s=4,
                           alpha=alpha)
        line, = ax.plot(ellipse[0, :] + mu[0], ellipse[1, :] + mu[1],
                        linestyle='-', linewidth=2, color=color, label=label,
                        alpha=alpha)
    else:
        line, point = artists
        point.set_offsets(mu)
        point.set_alpha(alpha)
        point.set_color(color)
        line.set_xdata(ellipse[0, :] + mu[0])
        line.set_ydata(ellipse[1, :] + mu[1])
        line.set_alpha(alpha)
        line.set_color(color)

    return (line, point) if point else (line,)

def plot_scores(allscores):
    # plot scores
    plt.figure()
    for scores in allscores:
        plt.plot(scores)
    plt.title('model vlb scores vs iteration')
    plt.show()

def plot_nMSE(all_nMSE):
    # plot nMSE
    plt.figure()
    for nMSE in all_nMSE:
        plt.plot(nMSE)
    plt.title('model nMSE vs iteration')
    plt.show()

def plot_absolute_error(all_err):
    # plot absolute error
    plt.figure()
    for err in all_err:
        plt.plot(err)
    plt.title('model absolute error vs iteration')
    plt.show()

def plot_prediction_2d(data, pred_y, data_label, save_prediction, visual_pdf_path, visual_tikz_path, legend_upper_right):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(data[:, 0], data[:, 1],c='black', s=1, zorder=1, label=data_label)
    plt.scatter(data[:, 0], pred_y, c='red', s=1, zorder=2, label='Prediction')

    if not legend_upper_right:
        legend = ax.legend(loc='lower left', markerscale=3)
    else:
        legend = ax.legend(loc='upper right', markerscale=3)
    ax.add_artist(legend)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # plt.title('Prediction fo)

    if save_prediction:
        tikz = os.path.join(visual_tikz_path + '_' + data_label + '.tex')
        pdf = os.path.join(visual_pdf_path +'_' +  data_label + '.pdf')
        tikzplotlib.get_tikz_code(figure=fig, filepath=None, figurewidth=None, figureheight=None, textsize=10.0,
                                  tex_relative_path_to_data=None, externalize_tables=False, override_externals=False,
                                  strict=False, wrap=True, add_axis_environment=True, extra_axis_parameters=None,
                                  extra_tikzpicture_parameters=None, dpi=None, show_info=False, include_disclaimer=True,
                                  standalone=False, float_format='{:.15g}', table_row_sep='\n')
        tikzplotlib.save(tikz, encoding=None)

        plt.savefig(pdf)

    plt.show()

def plot_prediction_2d_mean(data, mean_function, plus_2std_function, minus_2std_function, data_label, save_prediction, visual_pdf_path, visual_tikz_path, legend_upper_right):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(data[:, 0], data[:, 1], c='black', s=1, zorder=1, label=data_label)
    plt.scatter(data[:, 0], mean_function, c='red', s=1, zorder=2, label='Prediction')
    plt.scatter(data[:, 0], plus_2std_function, c='darksalmon', s=1, zorder=2, label='Confidence Interval')
    plt.scatter(data[:, 0], minus_2std_function, c='darksalmon', s=1, zorder=2)

    # plt.title('best model')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if not legend_upper_right:
        legend = ax.legend(loc='lower left', markerscale=3)
    else:
        legend = ax.legend(loc='upper right', markerscale=3)
    ax.add_artist(legend)

    if save_prediction:
        tikz = os.path.join(visual_tikz_path + '_' + data_label + '_conf.tex')
        pdf = os.path.join(visual_pdf_path + '_' + data_label + '_conf.pdf')
        tikzplotlib.get_tikz_code(figure=fig, filepath=None, figurewidth=None, figureheight=None, textsize=10.0,
                                  tex_relative_path_to_data=None, externalize_tables=False, override_externals=False,
                                  strict=False, wrap=True, add_axis_environment=True, extra_axis_parameters=None,
                                  extra_tikzpicture_parameters=None, dpi=None, show_info=False, include_disclaimer=True,
                                  standalone=False, float_format='{:.15g}', table_row_sep='\n')
        tikzplotlib.save(tikz, encoding=None)

        plt.savefig(pdf)

    plt.show()

def endeffector_pos_2d(data, in_dim_niw, pred_y, data_label, visual_pdf_path, visual_tikz_path, save_kinematics):
    # plot of prediction for endeffector positions vs. data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(data[:, in_dim_niw], data[:, in_dim_niw+1], s=3, label=data_label, color="black")
    plt.scatter(pred_y[:, 0], pred_y[:, 1], c='red', s=3, label='Prediction')
    if in_dim_niw > 1:
        plt.plot([data[:, in_dim_niw], pred_y[:, 0]], [data[:, in_dim_niw+1], pred_y[:, 1]],color="green",zorder=1)
    plt.title('X- and Y-Position of Endeffector')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    legend = ax.legend(loc='upper right')
    ax.add_artist(legend)

    if save_kinematics:
        tikz = os.path.join(visual_tikz_path + '_2d_' + data_label + '.tex')
        pdf = os.path.join(visual_pdf_path + '_2d_' + data_label + '.pdf')
        tikzplotlib.get_tikz_code(figure=fig, filepath=None, figurewidth=None, figureheight=None, textsize=10.0,
                                  tex_relative_path_to_data=None, externalize_tables=False, override_externals=False,
                                  strict=False, wrap=True, add_axis_environment=True, extra_axis_parameters=None,
                                  extra_tikzpicture_parameters=None, dpi=None, show_info=False, include_disclaimer=True,
                                  standalone=False, float_format='{:.15g}', table_row_sep='\n')
        tikzplotlib.save(tikz, encoding=None)

        plt.savefig(pdf)

    plt.show()

def endeffector_pos_3d(data, pred, in_dim_niw, data_label, visual_pdf_path, visual_tikz_path, save_kinematics):

    # plot of prediction for endeffector positions vs. data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = plt.axes(projection='3d')
    ax.scatter(data[:, 2], data[:, 1], data[:, 0], c='black', label=data_label, s=3, zorder = 3)
    ax.scatter(pred[:, 1], pred[:, 0], data[:, 0], c='red', label='Prediction', s=3, zorder=1)


    plt.title('X- and Y-Position of Endeffector \n over Joint Angle')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Joint Angle in rad')
    ax.tick_params(labelsize=10)

    # N =5
    # ymin, ymax = ax.get_ylim()
    # ax.set_yticks(np.round(np.linspace(ymin, ymax, N), 2))
    # xmin, xmax = ax.get_xlim()
    # ax.set_xticks(np.round(np.linspace(xmin, xmax, N), 2))

    # ax.view_init(azim=30)
    legend = ax.legend(loc='upper right', prop={'size': 12})
    ax.add_artist(legend)
    # ax.contour3D(data_test[:,0], data_test[:,1], data_test[:,2], 50, cmap='binary')
    # ax.contour3D(data_test[:,0], data_test[:,1], mean_function, 50, cmap='Greens')

    if save_kinematics:
        tikz = os.path.join(visual_tikz_path + '_3d_' + data_label + '.tex')
        pdf = os.path.join(visual_pdf_path + '_3d_' + data_label + '.pdf')
        tikzplotlib.get_tikz_code(figure=fig, filepath=None, figurewidth=None, figureheight=None, textsize=10.0,
                                  tex_relative_path_to_data=None, externalize_tables=False, override_externals=False,
                                  strict=False, wrap=True, add_axis_environment=True, extra_axis_parameters=None,
                                  extra_tikzpicture_parameters=None, dpi=None, show_info=False, include_disclaimer=True,
                                  standalone=False, float_format='{:.15g}', table_row_sep='\n')
        tikzplotlib.save(tikz, encoding=None)

        plt.savefig(pdf)

    plt.show()

# # plot of inverse dynamics of first joint: q,q_dot,q_dot_dot, motor torque and predicted motor torque
# if plot_dynamics == True:
#     f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
#
#     ax1.plot(np.arange(1,n_train+1), data[:,0])
#     ax2.plot(np.arange(1,n_train+1), data[:,1])
#     ax3.plot(np.arange(1,n_train+1), data[:,2])
#     ax4.plot(np.arange(1,n_train+1), data[:,in_dim_niw])
#     ax5.plot(np.arange(1,n_train+1), pred_y[:,0])
#
#     plt.show()

def motor_torque(n_train, data, pred_y, in_dim_niw, save_dynamics, visual_tikz_path, visual_pdf_path, data_label):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.plot(np.arange(1, n_train + 1), data[:, in_dim_niw], color="black", label=data_label)
    ax.plot(np.arange(1, n_train + 1), pred_y[:, 0], color="red", label='Prediction')
    # plt.title("Prediction for the torque of the first joint of Barret WAM (inverse dynamics data)")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Motor Torque", fontsize=12)

    legend = ax.legend(loc='upper right', markerscale=3)
    ax.add_artist(legend)

    if save_dynamics:
        tikz = os.path.join(visual_tikz_path + '_3d_' + data_label + '.tex')
        pdf = os.path.join(visual_pdf_path + '_3d_' + data_label + '.pdf')
        tikzplotlib.get_tikz_code(figure=fig, filepath=None, figurewidth=None, figureheight=None, textsize=10.0,
                                  tex_relative_path_to_data=None, externalize_tables=False, override_externals=False,
                                  strict=False, wrap=True, add_axis_environment=True, extra_axis_parameters=None,
                                  extra_tikzpicture_parameters=None, dpi=None, show_info=False, include_disclaimer=True,
                                  standalone=False, float_format='{:.15g}', table_row_sep='\n')
        tikzplotlib.save(tikz, encoding=None)

        plt.savefig(pdf)

    plt.show()

def violin_plot_two(data, num_columns=None, tikz_path=None, pdf_path=None, x_label=None, y_label=None, title=None, x_categories=None):
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel(x_label)

    # print(data)
    # print(data.shape)

    # create test data
    # np.random.seed(19680801)
    # data = [sorted(np.random.normal(0, std, 20)) for std in range(1, 5)]
    # data = np.asarray(data).T
    # print(data.shape)

    # print(data)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)

    ax1.set_title(title)
    ax1.set_ylabel(y_label)
    ax1.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    ax2.set_title(title)
    # data = np.ndarray.tolist(data)
    parts = ax2.violinplot(
        data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)#axis=1)
    if num_columns != 1:
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[0, :], whiskers[1, :] # switched from whiskers[:,0], whiskers[:, 1]
        inds = np.arange(1, len(medians) + 1)
    else:
        whiskers = np.array([data, quartile1, quartile3])
        whiskersMin, whiskersMax = whiskers[0], whiskers[1]
        inds = np.arange(1, 2)


    ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax2.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    # set style for the axes
    labels = x_categories

    for ax in [ax1, ax2]:
        set_axis_style(ax, labels)
        set_axis_style(ax, labels)

    # plt.subplots_adjust(bottom=0.15, wspace=0.05)
    tikzplotlib.get_tikz_code(figure=fig, filepath=None, figurewidth=None, figureheight=None, textsize=10.0,
                              tex_relative_path_to_data=None, externalize_tables=False, override_externals=False,
                              strict=False, wrap=True, add_axis_environment=True, extra_axis_parameters=None,
                              extra_tikzpicture_parameters=None, dpi=None, show_info=False, include_disclaimer=True,
                              standalone=False, float_format='{:.15g}', table_row_sep='\n')
    tikzplotlib.save(tikz_path, encoding=None)
    plt.savefig(pdf_path)
    plt.show()

def violin_plot_one(data, num_columns=None, tikz_path=None, pdf_path=None, x_label=None, y_label=None, title=None, x_categories=None):
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel(x_label)

    # print(data)
    # print(data.shape)

    # create test data
    # np.random.seed(19680801)
    # data = [sorted(np.random.normal(0, std, 20)) for std in range(1, 5)]
    # data = np.asarray(data).T
    # print(data.shape)

    # print(data)

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
    fig, ax2 = plt.subplots(nrows=1, ncols=1, sharey=True)

    # ax1.set_title(title)
    # ax1.set_ylabel(y_label)
    # ax1.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    # fig.set_title(title)
    # data = np.ndarray.tolist(data)
    parts = ax2.violinplot(
        data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        # pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)#axis=1)
    if num_columns != 1:
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[0, :], whiskers[1, :] # switched from whiskers[:,0], whiskers[:, 1]
        inds = np.arange(1, len(medians) + 1)
    else:
        whiskers = np.array([data, quartile1, quartile3])
        whiskersMin, whiskersMax = whiskers[0], whiskers[1]
        inds = np.arange(1, 2)


    ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax2.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    # set style for the axes
    labels = x_categories

    # for ax in [ax1, ax2]:
    #     set_axis_style(ax, labels)
    #     set_axis_style(ax, labels)

    # for ax in [ax1, ax2]:
    set_axis_style(ax2, labels)
    set_axis_style(ax2, labels)
    ax2.set_ylabel(y_label)

    # plt.subplots_adjust(bottom=0.15, wspace=0.05)
    tikzplotlib.get_tikz_code(figure=fig, filepath=None, figurewidth=None, figureheight=None, textsize=10.0,
                              tex_relative_path_to_data=None, externalize_tables=False, override_externals=False,
                              strict=False, wrap=True, add_axis_environment=True, extra_axis_parameters=None,
                              extra_tikzpicture_parameters=None, dpi=None, show_info=False, include_disclaimer=True,
                              standalone=False, float_format='{:.15g}', table_row_sep='\n')
    tikzplotlib.save(tikz_path, encoding=None)
    plt.savefig(pdf_path)
    plt.show()