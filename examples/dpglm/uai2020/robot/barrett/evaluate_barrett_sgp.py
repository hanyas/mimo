import os
import argparse

import numpy as np
import numpy.random as npr

import mimo

from reg import SparseGPListRegressor


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate GP')
    parser.add_argument('--datapath', help='path to dataset', default=os.path.abspath(mimo.__file__ + '/../../datasets'))
    parser.add_argument('--evalpath', help='path to evaluation', default=os.path.abspath(mimo.__file__ + '/../../evaluation/uai2020'))

    args = parser.parse_args()

    np.random.seed(1337)

    import scipy as sc
    from scipy import io

    # load all available data
    _data = sc.io.loadmat(args.datapath + '/Barrett/ias_real_barrett_data.mat')

    _train_data = np.hstack((_data['X_train'], _data['Y_train']))
    _test_data = np.hstack((_data['X_test'], _data['Y_test']))

    data = np.vstack((_train_data, _test_data))

    train_input = _train_data[:, :21]
    train_target = _train_data[:, 21:]

    test_input = _test_data[:, :21]
    test_target = _test_data[:, 21:]

    gp = SparseGPListRegressor(7, train_input, inducing_size=2500, device='gpu')
    gp.fit(train_target, train_input, nb_iter=150, lr=0.05, preprocess=True)

    test_predict = gp.predict(test_input)

    from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

    test_mse = mean_squared_error(test_target, test_predict)
    test_smse = 1. - r2_score(test_target, test_predict, multioutput='variance_weighted')
    test_evar = explained_variance_score(test_target, test_predict, multioutput='variance_weighted')

    print('TEST - MSE:', test_mse, 'SMSE:', test_smse, 'EVAR:', test_evar)

    arr = np.array([test_mse, test_smse, test_evar])
    np.savetxt('barrett_sgp.csv', arr, delimiter=',')
