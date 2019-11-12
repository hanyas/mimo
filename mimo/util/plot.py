import numpy as np
from matplotlib import pyplot as plt


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
        print(err)
        plt.plot(err)
    plt.title('model absolute error vs iteration')
    plt.show()

def plot_prediction_2d(data, pred_y):
    plt.scatter(data[:, 0], data[:, 1],c='black', s=1, zorder=1)
    plt.scatter(data[:, 0], pred_y, c='red', s=1, zorder=2)
    # axes = plt.gca()
    # axes.set_xlim([xmin, xmax])
    # axes.set_ylim([-5, 5])
    plt.title('best model')
    plt.savefig('results/_training.pdf')
    plt.show()

def plot_prediction_2d_mean(data, mean_function, plus_2std_function, minus_2std_function):
    plt.scatter(data[:, 0], data[:, 1], c='black', s=1, zorder=1)
    plt.scatter(data[:, 0], mean_function, c='red', s=1, zorder=2)
    plt.scatter(data[:, 0], plus_2std_function, c='darksalmon', s=0.5, zorder=2)
    plt.scatter(data[:, 0], minus_2std_function, c='darksalmon', s=0.5, zorder=2)
    # axes = plt.gca()
    # axes.set_xlim([xmin, xmax])
    # axes.set_ylim([-5, 5])
    plt.title('best model')
    plt.savefig('results/_testing.pdf')
    plt.show()

def endeffector_pos(data, in_dim_niw, pred_y, string):
    # plot of prediction for endeffector positions vs. data
    plt.scatter(data[:, in_dim_niw], data[:, in_dim_niw+1], s=1, zorder=2)
    plt.scatter(pred_y[:, 0], pred_y[:, 1], c='red', s=1, zorder=2)
    plt.plot([data[:, in_dim_niw], pred_y[:, 0]], [data[:, in_dim_niw+1], pred_y[:, 1]],color="green",zorder=1)
    plt.title('best model')
    plt.savefig(string)
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

def motor_torque(n_train, data, pred_y, in_dim_niw):
    plt.figure(figsize=(40, 20))
    plt.plot(np.arange(1, n_train + 1), data[:, in_dim_niw], color="blue", label='data')
    plt.plot(np.arange(1, n_train + 1), pred_y[:, 0], color="red", label='prediction')
    plt.title("Prediction for the torque of the first joint of Barret WAM (inverse dynamics data)")
    plt.xlabel("Time / Data Index")
    plt.ylabel("Torque")
    plt.savefig('inverse_dynamics.svg')
    plt.show()