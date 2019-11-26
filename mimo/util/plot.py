import numpy as np
import os

from matplotlib import pyplot as plt
import tikzplotlib


def plot_gaussian(mu, lmbda, color='r', label='', alpha=1.0, ax=None, artists=None):
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


def plot_violin_box(data, nb_cols=None,
                    tikz_path=None, pdf_path=None,
                    x_label=None, y_label=None,
                    title=None, x_categories=None,
                    violin=True, box=True, show=False):

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel(x_label)

    if box:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharey='all')
        ax1.boxplot(data, showfliers=True, showmeans=True, meanline=True)

        labels = x_categories
        set_axis_style(ax1, labels)
        set_axis_style(ax1, labels)
        ax1.set_ylabel(y_label)

        tikzplotlib.save(tikz_path + '_box.tex', encoding=None)
        plt.savefig(pdf_path + '_box.pdf')
        if show:
            plt.show()

    if violin:
        fig, ax2 = plt.subplots(nrows=1, ncols=1, sharey='all')
        parts = ax2.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
        if nb_cols != 1:
            inds = np.arange(1, len(medians) + 1)
        else:
            inds = np.arange(1, 2)

        ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)

        labels = x_categories
        set_axis_style(ax2, labels)
        set_axis_style(ax2, labels)
        ax2.set_ylabel(y_label)

        tikzplotlib.save(tikz_path + '_violin.tex', encoding=None)
        plt.savefig(pdf_path + '_violin.pdf')

        if show:
            plt.show()
