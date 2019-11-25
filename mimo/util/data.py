import numpy as np
import numpy.random as npr

from mimo import distributions
import csv
import os
import random


def load_data(n_train, n_test, keyword, dir, output_dim, input_dim, sarcos, seed=1337):
    # set random seed
    np.random.seed(seed=seed)

    os.chdir(dir)
    data = np.genfromtxt(keyword, dtype=None, encoding=None, delimiter=",")

    np.random.shuffle(data)

    # generate subset of training_data points
    train_data = data[:n_train, :]
    test_data = data[n_train:n_train + n_test, :]

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

        np.random.shuffle(test_data)
        test_data = test_data[:n_test, :]

    return train_data, test_data


def generate_linear(n_train, input_dim, output_dim, shuffle=False, seed=1337):
    # set random seed
    np.random.seed(seed=seed)

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
    np.random.seed(seed=seed)

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
    np.random.seed(seed=seed)

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


def generate_kinematics(n_train=None, output_dim=2, num_joints=None,
                        loc_noise=None, scale_noise=None,
                        l1=None, l2=None, l3=None, l4=None,
                        seed=1337, csv=False, filename=None):
    assert output_dim >= 2

    # set random seed
    np.random.seed(seed=seed)

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


def generate_noisey_heaviside(n_train, scaling=1., seed=1337):
    # set random seed
    np.random.seed(seed=seed)

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


def generate_gaussian(n_train, output_dim, input_dim, seed=1337):
    # set random seed
    np.random.seed(seed=seed)

    _A = 1. * npr.randn(output_dim, input_dim)
    dist = distributions.LinearGaussian(A=_A, sigma=25e-2 * np.eye(output_dim), affine=False)
    data = dist.rvs(size=n_train)
    return data


def generate_sarcos(n_train, n_test, input_dim, output_dim,
                    seed=None, all=False, shuffle=False):
    # set seed
    np.random.seed(seed=seed)

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
    np.random.seed(seed=seed)

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
