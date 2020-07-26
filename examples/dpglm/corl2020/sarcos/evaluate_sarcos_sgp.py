import os
import argparse

import numpy as np
import numpy.random as npr

import mimo

from reg import SparseGPRegressor


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate Sparse GP')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020'))

    args = parser.parse_args()

    np.random.seed(1337)

    import scipy as sc
    from scipy import io

    # load all available data
    _train_data = sc.io.loadmat(args.datapath + '/sarcos/sarcos_inv.mat')['sarcos_inv']
    _test_data = sc.io.loadmat(args.datapath + '/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

    train_input = _train_data[:, :21]
    train_target = _train_data[:, 21:]

    test_input = _test_data[:, :21]
    test_target = _test_data[:, 21:]

    # scale data
    input_data = np.vstack((train_input, test_input))
    target_data = np.vstack((train_target, test_target))

    from sklearn.preprocessing import StandardScaler
    input_transform = StandardScaler()
    target_transform = StandardScaler()

    input_transform.fit(input_data)

    from sklearn.metrics import mean_squared_error, r2_score

    nb_seeds = 5
    nb_outputs = train_target.shape[-1]

    test_mse, test_smse = [], []

    for n in range(nb_seeds):
        print('Training seed:', n)

        _train_mu = np.zeros((len(train_input), nb_outputs))
        _test_mu = np.zeros((len(test_input), nb_outputs))

        for i in range(nb_outputs):
            target_transform.fit(target_data[:, i][:, None])

            _gp = SparseGPRegressor(train_input, inducing_size=5000, device='gpu',
                                    input_transform=input_transform,
                                    target_transform=target_transform)

            _gp.fit(train_target[:, i], train_input, nb_iter=75, lr=0.05,
                    preprocess=True, verbose=False)

            _train_mu[:, i] = _gp.predict(train_input)
            _test_mu[:, i] = _gp.predict(test_input)

        _train_mse = mean_squared_error(train_target, _train_mu)
        _train_smse = 1. - r2_score(train_target, _train_mu)
        print('TRAIN - MSE:', _train_mse, 'SMSE:', _train_smse)

        _test_mse = mean_squared_error(test_target, _test_mu)
        _test_smse = 1. - r2_score(test_target, _test_mu)
        print('TEST - MSE:', _test_mse, 'SMSE:', _test_smse)

        test_mse.append(_test_mse)
        test_smse.append(_test_smse)

    mean_mse = np.mean(test_mse)
    std_mse = np.std(test_mse)

    mean_smse = np.mean(test_smse)
    std_smse = np.std(test_smse)

    arr = np.array([mean_mse, std_mse,
                    mean_smse, std_smse])

    import pandas as pd
    dt = pd.DataFrame(data=arr, index=['mse_avg', 'mse_std',
                                       'smse_avg', 'smse_std'])

    dt.to_csv('sarcos_sgp.csv', mode='a', index=True)
