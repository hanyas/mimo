import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from mimo import distributions

import random

def generate_LIN(n_train,in_dim_niw, out_dim, freq, shuffle=False, seed=None):
    # set seed
    np.random.seed(seed=seed)

    # create sin data
    data = np.zeros((n_train, in_dim_niw + out_dim))
    step = 1 / n_train
    for i in range(data.shape[0]):
        x = i * step
        data[i, 0] = (x + npr.normal(0, 0.1))
        data[i, 1] = x + 0.5
    if shuffle:
        np.random.shuffle(data)
    return data

def generate_CMB(n_train, seed=None):
    # set seed
    np.random.seed(seed=seed)

    # load Cosmic Microwave Background (CMB) training_data from Hannah (2011)
    data = np.genfromtxt("datasets/cmb.csv", dtype=None, encoding=None, usecols=(0, 1))
    np.random.shuffle(data)

    # generate subset of training_data points
    training_data = data[:n_train, :]
    test_data = data[n_train:, :]
    data = training_data
    # training_data = training_data[np.random.choice(training_data.shape[0], size=n_samples, replace=False), :]

    return data

def generate_SIN(n_train,in_dim_niw, out_dim, freq, shuffle=False, seed=None):
    # set seed
    np.random.seed(seed=seed)

    # create sin data
    data = np.zeros((n_train, in_dim_niw + out_dim))
    step = freq * np.pi / n_train
    for i in range(data.shape[0]):
        x = i * step - 6.
        data[i, 0] = (x + npr.normal(0, 0.1))
        data[i, 1] = (3. * (np.sin(x) + npr.normal(0, .1)))
    if shuffle:
        np.random.shuffle(data)
    return data

def generate_kinematics(n_train=None,out_dim=None,num_joints=None,loc_noise=None,scale_noise=None,l1=None,l2=None,l3=None,l4=None, seed=None):
    # set seed
    np.random.seed(seed=seed)
    # transform degree to rad
    scale_noise = scale_noise * 2 * np.pi / 360
    loc_noise = loc_noise * 2 * np.pi / 360

    # joint angles
    q = npr.uniform(low=np.zeros(num_joints),high=np.ones(num_joints),size=(n_train,num_joints)) * 2 * np.pi #+ npr.normal(loc_noise,scale_noise)
    q = q % 2*np.pi
    q1 = q[:,0]

    # position of end effector
    if out_dim == 1:
        return None
    elif out_dim == 2:
        pos_x = l1*np.cos(q1)
        pos_y = l1*np.sin(q1)
        if num_joints >= 2:
            q2 = q[:,1]
            pos_x = pos_x + l2*np.cos(q1+q2)
            pos_y = pos_y + l2*np.sin(q1+q2)
            if num_joints >= 3:
                q3 = q[:,2]
                pos_x = pos_x + l3*np.cos(q1+q2+q3)
                pos_y = pos_y + l3*np.sin(q1+q2+q3)
                if num_joints >= 4:
                    q4 = q[:,3]
                    pos_x = pos_x + l4 * np.cos(q1 + q2 + q3 + q4)
                    pos_y = pos_y + l4 * np.sin(q1 + q2 + q3 + q4)

    # concatenate data
    data = np.zeros((n_train,num_joints + out_dim))
    for i in range(n_train):
        for j in range(num_joints + 2):
            if j < num_joints:
                data[i,j] = q[i,j]
            elif j == num_joints:
                data[i,j] = pos_x[i]
            elif j == num_joints + 1:
                data[i,j] = pos_y[i]

    return data

def generate_kinematics_cos(n_train=None,out_dim=None,num_joints=None,loc_noise=None,scale_noise=None,l1=None,l2=None,l3=None):
    # transform degree to rad
    scale_noise = scale_noise * 2 * np.pi / 360
    loc_noise = loc_noise * 2 * np.pi / 360

    # joint angles
    q = npr.uniform(low=np.zeros(num_joints),high=np.ones(num_joints),size=(n_train,num_joints)) * 2 * np.pi #+ npr.normal(loc_noise,scale_noise)
    q = q % 2*np.pi
    q1 = np.cos(q[:,0])

    def cos_to_sin(cos):
        sin = np.sqrt(1-cos**2)
        return -sin

    # position of end effector
    if out_dim == 1:
        return None
    elif out_dim == 2:
        pos_x = l1*q1
        pos_y = l1*cos_to_sin(q1)
        # if num_joints >= 2:
        #     q2 = q[:,1]
        #     pos_x = pos_x + l2*np.cos(q1+q2)
        #     pos_y = pos_y + l2*np.sin(q1+q2)
        #     if num_joints == 3:
        #         q3 = q[:,2]
        #         pos_x = pos_x + l3*np.cos(q1+q2+q3)
        #         pos_y = pos_y + l3*np.sin(q1+q2+q3)

    # concatenate data
    data = np.zeros((n_train,num_joints + out_dim))
    for i in range(n_train):
        for j in range(num_joints + 2):
            if j < num_joints:
                data[i,j] = q[i,j]
            elif j == num_joints:
                data[i,j] = pos_x[i]
            elif j == num_joints + 1:
                data[i,j] = pos_y[i]

    return data

def generate_heaviside1(n_train):
    # create training_data from heaviside function
    def f(x):
        rnd_V = 0.5 * (np.sign(x) + 1)
        return rnd_V
    # xvals1 = sorted(np.concatenate([np.linspace(-1,-0.01,n_train),[0]]))
    # xvals2 = sorted(np.concatenate([np.linspace(0.01,1,n_train),[0]]))
    xvals1 = np.linspace(-1,-0.01,n_train / 2)
    xvals2 = np.linspace(0.01,1,n_train / 2 )
    xvals = np.concatenate((xvals1, xvals2))
    yvals = f(xvals)

    # xval_add = np.zeros(n_train)
    # yval_add = np.concatenate([np.linspace(0,1,n_train)])
    # for i in range(100):
    #     xval_add[i] = 0
    #     #yval_add[i] = 0.5
    # xvals = np.concatenate((xval, xval_add))
    # yvals = np.concatenate((yval, yval_add))
    training_data = np.zeros((len(xvals), 2))

    for i in range(len(xvals)):
        training_data[i, 0] = xvals[i]
        training_data[i, 1] = yvals[i]
    data = training_data
    return data

def generate_heaviside2(n_train, scaling=None):
    # create training_data from heaviside function
    def f(x,w):
        rnd_V = 0.5 * (np.sign(x) + 1) + w
        return rnd_V

    xvals = npr.normal(0,1,n_train) * scaling
    w = npr.normal(0,0.01,n_train) * scaling
    yvals = f(xvals,w)

    training_data = np.zeros((len(xvals), 2))
    for i in range(len(xvals)):
        training_data[i, 0] = xvals[i]
        training_data[i, 1] = yvals[i]
    data = training_data
    return data

def generate_gaussian(n_train, out_dim, in_dim_niw, seed):
    # set seed
    np.random.seed(seed=seed)

    _A = 1. * npr.randn(out_dim, in_dim_niw)
    dist = distributions.LinearGaussian(A=_A, sigma=2.5e-1 * np.eye(out_dim), affine=False)
    data = dist.rvs(size=n_train)
    return data

def generate_Sarcos(n_train,n_test,in_dim_niw,out_dim, seed=None, all=False):
    # set seed
    np.random.seed(seed=seed)

    # load Sarcos data from https://www.ias.informatik.tu-darmstadt.de/Miscellaneous/Miscellaneous
    if all == False:
        X = np.genfromtxt("datasets/sarcos/X_train.csv", dtype=None, encoding=None, usecols=(0, 1, 2), delimiter=",")
        y = np.genfromtxt("datasets/sarcos/Y_train.csv", dtype=None, encoding=None, usecols=(0), delimiter=",")
        y = y[..., np.newaxis]

        # create data array
        data = np.zeros((X.shape[0],X.shape[1]+out_dim))
        data[:,:-1] = X
        data[:,in_dim_niw:] = y

    elif all == True:
        X = np.genfromtxt("datasets/sarcos/X_train.csv", dtype=None, encoding=None, delimiter=",")
        y = np.genfromtxt("datasets/sarcos/Y_train.csv", dtype=None, encoding=None, delimiter=",")
        # y = y[..., np.newaxis]

        # create data array
        data = np.zeros((X.shape[0],X.shape[1]+out_dim))
        data[:,:-7] = X
        data[:,in_dim_niw:] = y

    # shuffle data randomly
    # np.random.shuffle(data)

    # generate subset of training_data points
    training_data = data[:n_train, :]
    test_data = data[n_train:, :]
    # training_data = training_data[np.random.choice(training_data.shape[0], size=n_samples, replace=False), :]

    return training_data

def generate_Barret(n_train,n_test,in_dim_niw,out_dim, seed=None, all=False):
    # set seed
    np.random.seed(seed=seed)

    # load Sarcos data from https://www.ias.informatik.tu-darmstadt.de/Miscellaneous/Miscellaneous
    if all == False:
        X = np.genfromtxt("datasets/Barrett/X_train.csv", dtype=None, encoding=None, usecols=(0, 1, 2), delimiter=",")
        y = np.genfromtxt("datasets/Barrett/Y_train.csv", dtype=None, encoding=None, usecols=(0), delimiter=",")
        y = y[..., np.newaxis]

        # create data array
        data = np.zeros((X.shape[0],X.shape[1]+out_dim))
        data[:,:-1] = X
        data[:,in_dim_niw:] = y

    elif all == True:
        X = np.genfromtxt("datasets/Barrett/X_train.csv", dtype=None, encoding=None, delimiter=",")
        y = np.genfromtxt("datasets/Barrett/Y_train.csv", dtype=None, encoding=None, delimiter=",")
        # y = y[..., np.newaxis]

        # create data array
        data = np.zeros((X.shape[0],X.shape[1]+out_dim))
        data[:,:-7] = X
        data[:,in_dim_niw:] = y

    # shuffle data randomly
    # np.random.shuffle(data)

    # generate subset of training_data points
    training_data = data[:n_train, :]
    test_data = data[n_train:, :]
    # training_data = training_data[np.random.choice(training_data.shape[0], size=n_samples, replace=False), :]

    return training_data

def normalize_data(data, scaling):
    # Normalize data to 0 mean, 1 std_deviation, optionally scale data
    mean = np.mean(data, axis=0)
    std_deviation = np.std(data,axis=0)
    data = (data - mean) / (std_deviation * scaling)
    return data

def center_data(data, scaling):
    # Center data to 0 mean
    mean = np.mean(data, axis=0)
    data = (data - mean) / scaling
    return data



