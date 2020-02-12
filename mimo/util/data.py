import numpy as np
import numpy.random as npr

import os
import random
import csv

from mimo import distributions


def sample_env(env, nb_rollouts, nb_steps,
               ctl=None, noise_std=0.1,
               apply_limit=True):
    obs, act = [], []

    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    ulim = env.action_space.high

    for n in range(nb_rollouts):
        _obs = np.zeros((nb_steps, dm_obs))
        _act = np.zeros((nb_steps, dm_act))

        x = env.reset()

        for t in range(nb_steps):
            if ctl is None:
                # unifrom distribution
                u = np.random.uniform(-ulim, ulim)
            else:
                u = ctl(x)
                u = u + noise_std * npr.randn(1, )

            if apply_limit:
                u = np.clip(u, -ulim, ulim)

            _obs[t, :] = x
            _act[t, :] = u

            x, r, _, _ = env.step(u)

        obs.append(_obs)
        act.append(_act)

    return obs, act


def load_data(n_train, n_test, keyword, dir, output_dim, input_dim, sarcos, seed=1337):
    # set random seed
    np.random.seed(seed)

    os.chdir(dir)
    data = np.genfromtxt(keyword, dtype=None, encoding=None, delimiter=",")

    # randomly create a training and a test set from the data, but keep ordering of data
    sample_size = n_train
    train_data = [data[i] for i in sorted(random.sample(range(len(data)), sample_size))]
    sample_size = n_test
    test_data = [data[i] for i in sorted(random.sample(range(len(data)), sample_size))]
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    # this is for traing and test sets with random ordering
    # np.random.shuffle(data)
    # generate subset of training_data points
    # train_data = data[:n_train, :]
    # test_data = data[n_train:n_train + n_test, :]

    if sarcos:
        X_train = np.genfromtxt("Sarcos/X_train.csv", dtype=None, encoding=None, delimiter=",")
        y_train = np.genfromtxt("Sarcos/Y_train.csv", dtype=None, encoding=None, delimiter=",")
        X_test = np.genfromtxt("Sarcos/X_test.csv", dtype=None, encoding=None, delimiter=",")
        y_test = np.genfromtxt("Sarcos/Y_test.csv", dtype=None, encoding=None, delimiter=",")

        train_data = np.zeros((n_train, X_train.shape[1] + output_dim))
        train_data[:n_train, :-7] = X_train[:n_train, :]
        train_data[:n_train, input_dim:] = y_train[:n_train, :]

        test_data = np.zeros((X_test.shape[0], X_test.shape[1] + output_dim))
        test_data[:, :-7] = X_test
        test_data[:, input_dim:] = y_test

        # np.random.shuffle(test_data) #Fixme
        test_data = test_data[:n_test, :]

    return train_data, test_data


# transform dataset to trajectory dataset: tuples (t_i,y_i) -> (y_t, y_t+1 - y_t)
def trajectory_data(data, output_dim, input_dim, traj_trick):

    n_train = len(data[:,0])
    data_new = np.zeros((n_train-1, output_dim + output_dim))

    X = data[:, :-output_dim]
    Y = data[:, input_dim:]
    if traj_trick:
        Y_diff = data[1:, input_dim:] - data[:-1, input_dim:]
    else:
        Y_diff = data[1:, input_dim:]

    data_new[:, :output_dim], data_new[:, output_dim:] = Y[:-1, :], Y_diff[:, :]

    return data_new


def generate_linear(n_train, input_dim, output_dim, shuffle=False, seed=1337):
    # set random seed
    np.random.seed(seed)

    data = np.zeros((n_train, input_dim + output_dim))
    step = 1 / n_train
    for i in range(data.shape[0]):
        x = i * step
        data[i, 0] = (x + npr.normal(0, 0.1))
        data[i, 1] = x + 0.5
    if shuffle:
        np.random.shuffle(data)
    return data


def generate_cmb(seed=1337, shuffle=True, csv=False):
    # set random seed
    np.random.seed(seed)

    # load Cosmic Microwave Background (CMB) training_data from Hannah (2011)
    data = np.genfromtxt("datasets/cmb.csv", dtype=None, encoding=None, usecols=(0, 1))
    if shuffle:
        np.random.shuffle(data)

    if csv:
        with open('cmb.csv', 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)
        csvFile.close()

    return data


def generate_sine(n_train, input_dim, output_dim, freq,
                  shuffle=False, seed=1337, csv=False):
    # set random seed
    np.random.seed(seed)

    # create sin data
    data = np.zeros((n_train, input_dim + output_dim))
    step = freq * np.pi / n_train
    for i in range(data.shape[0]):
        x = i * step
        data[i, 0] = (x + npr.normal(0, 0.1))
        data[i, 1] = (3. * (np.sin(x) + npr.normal(0, .1)))
    if shuffle:
        np.random.shuffle(data)

    if csv:
        with open('sine.csv', 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)
        csvFile.close()

    return data


def generate_noisy_sine(n_train, input_dim, output_dim):

    data = np.zeros((n_train, input_dim + output_dim))
    xvals = np.linspace(-6 * np.pi, 6*np.pi, n_train)

    def sigmoid_func(x):
        f = 1 / (1 + np.exp(-x))
        return f

    def sigmoid_func_plus1(x):
        f = 1 / (1 + np.exp(-x))
        f = f + 0.5
        return f

    noise_std = np.zeros(data.shape[0])

    for i in range(data.shape[0]):
        x = xvals[i]
        data[i, 0] = x

        noise_std[i] = (0.75 * sigmoid_func_plus1(x - 8) + 0.25 - 0.40 * sigmoid_func(x + 10))

        # noise_std = 0.5 * np.exp(0.5 * np.sin(2*np.pi*x))
        # noise_std = 0.1*x

        w = npr.normal(0, noise_std[i])
        data[i, 1] = np.sin(0.3*x) + w

    # plt.scatter(xvals, noise_std)
    # plt.show()

        # if data[i, 0] < np.pi / 2:
        #     data[i, 1] = np.sin(x) + npr.normal(0, 0.3)
        #
        # if data[i, 0] >= np.pi / 2 and data[i, 0] <  np.pi:
        #     data[i, 1] = np.sin(x) + npr.normal(0, 0.1)
        #
        # if data[i, 0] >= np.pi and data[i, 0] < 3/2 * np.pi:
        #     data[i, 1] = np.sin(x) + npr.normal(0, 0.3)
        #
        # if data[i, 0] >= 3 / 2 * np.pi and data[i, 0] <= 2 * np.pi:
        #     data[i, 1] = np.sin(x) + npr.normal(0, 0.1)

    with open('sine_noise_sigmoids.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    csvFile.close()

    return data


def noise_function(n_train, noise_std1, noise_std2, xrange):

    xvals = np.linspace(xrange[0], xrange[1], n_train)
    # xvals = np.linspace(-10 * np.pi, 10 * np.pi, n_train)
    noise = np.zeros((n_train, 2))
    # noise = np.zeros(n_train)

    for i in range(noise.shape[0]):
        x = xvals[i]
        noise[i, 0] = x

        noise[i, 1] = 0.1

        # # FOR SINE_NOISE_SIGMOID
        # def sigmoid_func(x):
        #     f = 1 / (1 + np.exp(-x))
        #     return f
        # def sigmoid_func_plus1(x):
        #     f = 1 / (1 + np.exp(-x))
        #     f = f + 0.5
        #     return f
        # noise[i, 1] = (0.75 * sigmoid_func_plus1(x - 8) + 0.25 - 0.40 * sigmoid_func(x + 10))

        # FOR SINE_NOISE_LINEAR
        # noise[i, 1] = 0.1 * x

        # FOR SINE_NOISE_EXPSIN
        # noise[i, 1] = 0.5 * np.exp(0.5 * np.sin(np.pi * x))

        # FOR SINE STEP
        # if noise[i, 0] < np.pi / 2:
        #     noise[i, 1] = noise_std2
        #
        # if noise[i, 0] >= np.pi / 2 and x < 3/2 * np.pi:
        #     noise[i, 1] = noise_std1
        #
        # # if data[i, 0] >= np.pi and data[i, 0] < 3/2 * np.pi:
        # #     data[i, 1] = np.sin(x) + npr.normal(0, 0.3)
        #
        # if noise[i, 0] >= 3 / 2 * np.pi and x <= 2 * np.pi:
        #     noise[i, 1] = noise_std2

    sorting = np.argsort(noise, axis=0)  # sort based on input values
    sorted_xvals = np.take_along_axis(noise[:, 0], sorting[:, 0], axis=0)
    sorted_noise = np.take_along_axis(noise[:, 1], sorting[:, 0], axis=0)

    return sorted_xvals, sorted_noise


def generate_kinematics(n_train=None, output_dim=2, num_joints=None,
                        loc_noise=None, scale_noise=None,
                        l1=None, l2=None, l3=None, l4=None,
                        seed=1337, csv=False, filename=None):
    assert output_dim >= 2

    # set random seed
    np.random.seed(seed)

    # transform degree to rad
    scale_noise = scale_noise * 2. * np.pi / 360.
    loc_noise = loc_noise * 2. * np.pi / 360.

    # joint angles
    q = npr.uniform(low=np.zeros(num_joints),
                    high=np.ones(num_joints),
                    size=(n_train, num_joints)) * 2. * np.pi  # + npr.normal(loc_noise, scale_noise)
    q = q % 2. * np.pi
    q1 = q[:, 0]

    # position of end effector
    pos_x = l1 * np.cos(q1)
    pos_y = l1 * np.sin(q1)
    if num_joints >= 2:
        q2 = q[:, 1]
        pos_x = pos_x + l2 * np.cos(q1 + q2)
        pos_y = pos_y + l2 * np.sin(q1 + q2)
        if num_joints >= 3:
            q3 = q[:, 2]
            pos_x = pos_x + l3 * np.cos(q1 + q2 + q3)
            pos_y = pos_y + l3 * np.sin(q1 + q2 + q3)
            if num_joints >= 4:
                q4 = q[:, 3]
                pos_x = pos_x + l4 * np.cos(q1 + q2 + q3 + q4)
                pos_y = pos_y + l4 * np.sin(q1 + q2 + q3 + q4)

    # concatenate data
    data = np.zeros((n_train, num_joints + output_dim))
    for i in range(n_train):
        for j in range(num_joints + 2):
            if j < num_joints:
                data[i, j] = q[i, j]
            elif j == num_joints:
                data[i, j] = pos_x[i]
            elif j == num_joints + 1:
                data[i, j] = pos_y[i]

    if csv and filename is not None:
        with open(filename, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)
        csvFile.close()

    return data


def generate_heaviside(n_train):
    # create data from heaviside function
    def f(x):
        y = 0.5 * (np.sign(x) + 1)
        return y

    xvals1 = np.linspace(-1, -0.01, n_train / 2)
    xvals2 = np.linspace(0.01, 1, n_train / 2)
    xvals = np.concatenate((xvals1, xvals2))
    yvals = f(xvals)

    data = np.zeros((len(xvals), 2))
    for i in range(len(xvals)):
        data[i, 0] = xvals[i]
        data[i, 1] = yvals[i]

    return data


def generate_noisy_heaviside(n_train, scaling=1., seed=1337):
    # set random seed
    np.random.seed(seed)

    # create data from heaviside function
    def f(x, w):
        y = 0.5 * (np.sign(x) + 1) + w
        return y

    xvals = npr.normal(0, 1, n_train) * scaling
    w = npr.normal(0, 0.01, n_train) * scaling
    yvals = f(xvals, w)

    data = np.zeros((len(xvals), 2))
    for i in range(len(xvals)):
        data[i, 0] = xvals[i]
        data[i, 1] = yvals[i]

    return data

def generate_noisy_step(n_train, scaling=1., seed=1337):
    # set random seed
    np.random.seed(seed)

    # create data from heaviside function
    def f(x, w):
        y = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] < 0:
                # y = 0.5 * (np.sign(x) + 1) + w
                y[i] = 0.5 * x[i] ** 3 + 0 * x[i] ** 2 + 0 * x[i] ** 1 + w[i]
            elif x[i] > 0:
                y[i] = 0.5 * x[i] ** 3 + 0 * x[i] ** 2 + 0 * x[i] ** 1 + w[i] + 15
        return y

    # xvals = npr.uniform(-1, 1, n_train) * scaling
    xvals = np.linspace(-4,4,n_train)
    w = 0 #npr.normal(0, 1, n_train) * scaling
    yvals = f(xvals, w)

    data = np.zeros((len(xvals), 2))
    for i in range(len(xvals)):
        data[i, 0] = xvals[i]
        data[i, 1] = yvals[i]

    with open('step_polynomial_deg3_v7_nonoise.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    csvFile.close()

    return data


def generate_gaussian(n_train, output_dim, input_dim, seed=1337):
    # set random seed
    np.random.seed(seed)

    _A = 1. * npr.randn(output_dim, input_dim)
    dist = distributions.LinearGaussian(A=_A, sigma=25e-2 * np.eye(output_dim), affine=False)
    data = dist.rvs(size=n_train)
    return data


def generate_sarcos(n_train, n_test, input_dim, output_dim,
                    seed=None, all=False, shuffle=False):
    # set seed
    np.random.seed(seed)

    # load Sarcos data from https://www.ias.informatik.tu-darmstadt.de/Miscellaneous/Miscellaneous
    if not all:
        X = np.genfromtxt("datasets/Sarcos/X_train.csv", dtype=None, encoding=None, usecols=(0, 1, 2), delimiter=",")
        y = np.genfromtxt("datasets/Sarcos/Y_train.csv", dtype=None, encoding=None, usecols=(0, ), delimiter=",")
        y = y[..., np.newaxis]

        # create data array
        data = np.zeros((X.shape[0], X.shape[1] + output_dim))
        data[:, :-1] = X
        data[:, input_dim:] = y
    else:
        X = np.genfromtxt("datasets/Sarcos/X_train.csv", dtype=None, encoding=None, delimiter=",")
        y = np.genfromtxt("datasets/Sarcos/Y_train.csv", dtype=None, encoding=None, delimiter=",")

        # create data array
        data = np.zeros((X.shape[0], X.shape[1] + output_dim))
        data[:, :-7] = X
        data[:, input_dim:] = y

    if shuffle:
        np.random.shuffle(data)

    return data


def generate_barrett(n_train, n_test, input_dim, output_dim,
                    seed=None, all=False, shuffle=False):
    # set seed
    np.random.seed(seed)

    # load Sarcos data from https://www.ias.informatik.tu-darmstadt.de/Miscellaneous/Miscellaneous
    if not all:
        X = np.genfromtxt("datasets/Barrett/X_train.csv", dtype=None, encoding=None, usecols=(0, 1, 2), delimiter=",")
        y = np.genfromtxt("datasets/Barrett/Y_train.csv", dtype=None, encoding=None, usecols=(0,), delimiter=",")
        y = y[..., np.newaxis]

        # create data array
        data = np.zeros((X.shape[0], X.shape[1] + output_dim))
        data[:, :-1] = X
        data[:, input_dim:] = y
    else:
        X = np.genfromtxt("datasets/Sarcos/X_train.csv", dtype=None, encoding=None, delimiter=",")
        y = np.genfromtxt("datasets/Sarcos/Y_train.csv", dtype=None, encoding=None, delimiter=",")

        # create data array
        data = np.zeros((X.shape[0], X.shape[1] + output_dim))
        data[:, :-7] = X
        data[:, input_dim:] = y

    if shuffle:
        np.random.shuffle(data)

    return data

def generate_goldberg(n_train, input_dim, output_dim):

    data = np.zeros((n_train, input_dim + output_dim))

    xvals = np.linspace(0, 1, n_train)

    for i in range(n_train):

        scale = 0.5 + xvals[i]
        noise = npr.normal(0, scale, 1)

        yval = 2 * np.sin(2 * np.pi * (xvals[i])) + noise

        data[i, 0] = xvals[i]
        data[i, 1] = yval

    with open('goldberg.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    csvFile.close()

    return data


def normalize_data(data, scaling):
    # Normalize data to 0 mean, 1 std_deviation, optionally scale data
    mean = np.mean(data, axis=0)
    std_deviation = np.std(data, axis=0)
    data = (data - mean) / (std_deviation * scaling)
    return data


def center_data(data, scaling):
    # Center data to 0 mean
    mean = np.mean(data, axis=0)
    data = (data - mean) / scaling
    return data
