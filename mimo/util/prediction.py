import numpy as np

def predict_train(gmm, data, out_dim, in_dim_niw):
    pred_y = np.zeros((data.shape[0], out_dim))
    err_squared = 0
    err = 0
    for i in range(data.shape[0]):
        idx = gmm.labels_list[0].z[i]
        x = data[i, :-out_dim]
        y = data[i, in_dim_niw:]
        pred_y[i] = gmm.components[idx].predict(x)
        err = err + np.absolute((y - pred_y[i]))
        err_squared = err_squared + ((y - pred_y[i]) ** 2)
    return pred_y, err, err_squared

def get_natural_parameters():
    return None

def student_t():
    return None

def matrix_t_parameters():
    return None

def mean_function():
    return None

def std_deviation_envelope():
    return None

