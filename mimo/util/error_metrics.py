import numpy as np

def calc_error_metrics(data, n, in_dim_niw, err, err_squared):

    n_train = n

    var = np.var(data[:, in_dim_niw:], axis=0)
    std = np.std(data[:, in_dim_niw:], axis=0)
    mean = np.mean(data[:, in_dim_niw:], axis=0)
    # print('var_targets',var)
    # print('std_targets',std)
    # print('mean_targets',mean)

    MSE = 1 / n_train * err_squared
    nMSE = MSE / var

    RMSE = np.sqrt(MSE)
    # nRMSE = RMSE / mean
    nRMSE = RMSE / std

    MAE = 1 / n_train * err
    nMAE = MAE / var
    # RMAE = np.sqrt(MAE)
    # nRMAE = RMAE / var

    # print('........................ERROR METRICS.........................')
    # print('MAE', MAE)
    # print('nMAE', nMAE)
    # # print('RMAE',RMAE)
    # # print('nRMAE',nRMAE)
    # print('..............................................................')
    # print('MSE', MSE)
    # print('nMSE', nMSE)
    # print('..............................................................')
    # print('RMSE', RMSE)
    # print('nRMSE', nRMSE)
    # print('..............................................................')

    return nMSE