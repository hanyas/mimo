import numpy as np
from mimo.util.prediction import get_component_standard_parameters
import os
import pandas

def calculate_mean_gibbs(dpglm, data_test, n_test, iter, dir):
    for i in range(n_test):
        for j in range(iter):
            if i == 0:
                dpglm.resample_model()
            for idx, c in enumerate(dpglm.components):
                if idx in dpglm.used_labels and i == 0:
                    record = get_component_standard_parameters(c.posterior)
                    path = os.path.join(dir + '\\examples\\dpglm\\mean_gibbs\\record'+ '_' + str(idx) + '_.csv')
                    f = open(path, 'a+')
                    f.write(str(record[0]) + str(record[1]) + str(record[2]) + str(record[3]) + str(record[4]) + str(record[5]) + str(record[6]) + str(record[7]) +"\n")
                    f.close()
            for idx, c in enumerate(dpglm.components):
                if idx in dpglm.used_labels:
                    path = os.path.join(dir + '\\examples\\dpglm\\mean_gibbs\\record' + '_' + str(idx) + '_.csv')
                    parameters = pandas.read_csv(path, header=None, dtype=None, engine='python', sep=" ",index_col=False).values #, usecols=[0]
                    print(parameters)