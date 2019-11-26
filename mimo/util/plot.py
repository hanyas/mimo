import numpy as np
import os

from matplotlib import pyplot as plt
import tikzplotlib


def plot_gaussian(mu, lmbda, color='rnd_psi_mniw', label='', alpha=1.0, ax=None, artists=None):
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


def violin_plot(data, num_columns=None, tikz_path=None, pdf_path=None, x_label=None, y_label=None, title=None, x_categories=None):
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

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
    fig, ax2 = plt.subplots(nrows=1, ncols=1, sharey=True)

    # ax1.set_title(title)
    # ax1.set_ylabel(y_label)
    # ax1.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    # fig.set_title(title)
    # data = np.ndarray.tolist(data)

    # ax2.boxplot(data , showfliers = True, showmeans = True, meanline = True)#,  whis=[5, 95])

    # UNCOMMENT FOR VIOLIN PLOT
    parts = ax2.violinplot( data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        # pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)  # axis=1
    if num_columns != 1:
        whiskers = np.array([adjacent_values(sorted_array, q1, q3)
                             for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[0, :], whiskers[1, :]  # switched from whiskers[:,0], whiskers[:, 1]
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
