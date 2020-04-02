import numpy as np

import scipy as sc
from scipy import stats
from scipy.stats import multivariate_normal as mvn

from mimo.util.general import matrix_linear_studentt
from mimo.util.general import matrix_linear_gaussian
from mimo.util.general import multivariate_t_loglik as mvt_logpdf

import pathos
from pathos.pools import _ProcessPool as Pool
nb_cores = pathos.multiprocessing.cpu_count()

import time


def kstep_error(dpglm, query, exogenous=None, horizon=1,
                prediction='average', incremental=True,
                input_scaler=None, target_scaler=None):

    start = time.clock()

    from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

    mse, smse, evar = [], [], []
    for _query, _exo in zip(query, exogenous):
        _query_list, _exo_list,  = [], []

        nb_steps = _query.shape[0] - horizon
        for t in range(nb_steps):
            _query_list.append(_query[t, :])
            _exo_list.append(_exo[t: t + horizon, :])

        _output = parallel_meanfield_forcast(dpglm, _query_list,
                                             _exo_list, horizon,
                                             prediction, incremental,
                                             input_scaler, target_scaler)

        target, output = [], []
        for t in range(nb_steps):
            target.append(_query[t + horizon, :])
            output.append(_output[t][-1, :])

        target = np.vstack(target)
        output = np.vstack(output)

        _mse = mean_squared_error(target, output)
        _smse = 1. - r2_score(target, output, multioutput='variance_weighted')
        _evar = explained_variance_score(target, output, multioutput='variance_weighted')

        mse.append(_mse)
        smse.append(_smse)
        evar.append(_evar)

    finish = time.clock()
    return np.mean(mse), np.mean(smse), np.mean(evar), finish - start


def parallel_meanfield_forcast(dpglm, query, exogenous=None, horizon=1,
                               prediction='average', incremental=True,
                               input_scaler=None, target_scaler=None):

    assert isinstance(query, list)

    def _loop(kwargs):
        return meanfield_forcast(dpglm, kwargs['query'],
                                 kwargs['exogenous'], horizon,
                                 prediction, incremental,
                                 input_scaler, target_scaler)

    nb_traj = len(query)

    kwargs_list = []
    for n in range(nb_traj):
        _query = query[n, :]
        _exogenous = None if exogenous is None else exogenous[n, :]
        kwargs = {'query': query, 'exogenous': exogenous}
        kwargs_list.append(kwargs)

    with Pool(processes=nb_cores) as p:
        res = p.map(_loop, kwargs_list, chunksize=int(nb_traj / nb_cores))

    return res


def meanfield_forcast(dpglm, query, exogenous=None, horizon=1,
                      prediction='average', incremental=True,
                      input_scaler=None, target_scaler=None):

    if exogenous is not None:
        assert horizon <= len(exogenous)

    _output = query
    output = [query]
    for h in range(horizon):
        _query = _output
        if exogenous is not None:
            _query = np.hstack((_query, exogenous[h, :]))

        _output, _, _, _ = meanfield_prediction(dpglm, _query, None,
                                                prediction, incremental,
                                                input_scaler, target_scaler)
        output.append(_output)

    return np.vstack(output)


def parallel_meanfield_prediction(dpglm, query, target=None,
                                  prediction='average', incremental=False,
                                  input_scaler=None, target_scaler=None):

    def _loop(kwargs):
        return meanfield_prediction(dpglm, kwargs['x'], kwargs['y'],
                                    prediction, incremental,
                                    input_scaler, target_scaler)

    nb_data = len(query)

    kwargs_list = []
    for n in range(nb_data):
        _query = query[n, :]
        _target = None if target is None else target[n, :]
        kwargs = {'x': _query, 'y': _target}
        kwargs_list.append(kwargs)

    with Pool(processes=nb_cores) as p:
        res = p.map(_loop, kwargs_list, chunksize=int(nb_data / nb_cores))

    mean, var, std, nlpd = list(map(np.vstack, zip(*res)))
    return mean, var, std, nlpd


def meanfield_gating(dpglm, input):
    nb_models = len(dpglm.components)

    # compute posterior mixing weights
    weights = dpglm.gating.posterior.mean()

    # calculate the marginal likelihood of query for each cluster
    # calculate the normalization term for mean function for query
    marginal_likelihood = np.zeros((nb_models, ))
    effective_weights = np.zeros((nb_models, ))

    for idx, c in enumerate(dpglm.components):
        # if idx in dpglm.used_labels:
        marginal_likelihood[idx] = np.exp(c.log_marginal_likelihood(input))
        effective_weights[idx] = weights[idx] * marginal_likelihood[idx]

    effective_weights = effective_weights / np.sum(effective_weights)

    return effective_weights


def meanfield_predictive_component(component, query, type='student-t'):
    affine = component.posterior.mniw.affine
    params = component.posterior.mniw.params

    if type == 'student-t':
        # predictive matrix student-t
        mu, sigma, df = matrix_linear_studentt(query, *params, affine)
    elif type == 'gaussian':
        # predictive matrix gaussian
        mu, sigma, df = matrix_linear_gaussian(query, *params, affine)
    else:
        raise NotImplementedError

    return mu, sigma, df


def meanfield_prediction(dpglm, query, target=None,
                         prediction='average', incremental=False,
                         input_scaler=None, target_scaler=None,
                         type='gaussian'):

    nb_dim = dpglm.components[0].drow

    if input_scaler is not None:
        assert target_scaler is not None
    if target_scaler is not None:
        assert input_scaler is not None

    if input_scaler is not None and target_scaler is not None:
        input = np.squeeze(input_scaler.transform(np.atleast_2d(query)))
    else:
        input = query

    if target is not None and target_scaler is not None:
        target = np.squeeze(target_scaler.transform(np.atleast_2d(target)))

    mean, variance, stdv, nlpd =\
        np.zeros((nb_dim, )), np.zeros((nb_dim, )), np.zeros((nb_dim, )), 0.

    weights = meanfield_gating(dpglm, input)

    if prediction == 'mode':
        mode = np.argmax(weights)
        mean, _sigma, df = meanfield_predictive_component(dpglm.components[mode], input, type)

        if type == 'student-t':
            # consider only diagonal variances for plots
            variance = np.diag(_sigma * df / (df - 2))
            if target is not None:
                nlpd = np.exp(mvt_logpdf(target, mean, _sigma, df))
        elif type == 'gaussian':
            # consider only diagonal variances for plots
            variance = np.diag(_sigma)
            if target is not None:
                nlpd = mvn(mean, _sigma).pdf(target)
        else:
            raise NotImplementedError

    elif prediction == 'average':
        for idx, comp in enumerate(dpglm.components):
            # if idx in dpglm.used_labels:
            _mu, _sigma, _df = meanfield_predictive_component(comp, input, type)

            if type == 'student-t':
                # consider only diagonal variances for plots
                _var = np.diag(_sigma * _df / (_df - 2))
                if target is not None:
                    nlpd += weights[idx] * np.exp(mvt_logpdf(target, _mu, _sigma, _df))
            elif type == 'gaussian':
                # consider only diagonal variances for plots
                _var = np.diag(_sigma)
                if target is not None:
                    nlpd += weights[idx] * mvn(_mu, _sigma).pdf(target)
            else:
                raise NotImplementedError

            # Mean of a mixture = sum of weighted means
            mean += _mu * weights[idx]

            # Variance of a mixture = sum of weighted variances + ...
            # ... + sum of weighted squared means - squared sum of weighted means
            variance += (_var + _mu**2) * weights[idx]
        variance -= mean**2

    if target is not None:
        nlpd = -1.0 * np.log(nlpd)
    else:
        nlpd = None

    if target_scaler is not None:
        mean = np.squeeze(target_scaler.inverse_transform(np.atleast_2d(mean)))
        trans = (np.sqrt(target_scaler.explained_variance_[:, None]) * target_scaler.components_).T
        variance = np.diag(trans.T @ np.diag(variance) @ trans)

    # only diagonal elements
    stdv = np.sqrt(variance)

    if incremental:
        mean = query[:nb_dim] + mean

    return mean, variance, stdv, nlpd


# Weighted EM predictions over all models
def em_prediction(dpglm, query):
    nb_models = len(dpglm.components)
    nb_dim = dpglm.components[0].drow

    # initialize variables
    mean, var, std = np.zeros((nb_dim, )), np.zeros((nb_dim, )), np.zeros((nb_dim, ))

    # mixing weights
    weights = dpglm.gating.probs

    # calculate the marginal likelihood of test data query for each cluster
    # calculate the normalization term for mean function for query
    normalizer = 0.
    lklhd = np.zeros((nb_models,))
    effective_weight = np.zeros((nb_models,))
    for idx, c in enumerate(dpglm.components):
        if idx in dpglm.used_labels:
            lklhd[idx] = np.exp(c.gaussian.log_likelihood(query))
            effective_weight[idx] = weights[idx] * lklhd[idx]
            normalizer = normalizer + weights[idx] * lklhd[idx]

    # calculate contribution of each cluster to mean function
    for idx, c in enumerate(dpglm.components):
        if idx in dpglm.used_labels:
            t_mean = c.predict(query)
            # Mean of a mixture = sum of weighted means
            mean += t_mean * effective_weight[idx] / normalizer

            # Variance of a mixture = sum of weighted variances + ...
            # ... + sum of weighted squared means - squared sum of weighted means
            t_var = np.diag(c.sigma)  # consider only diagonal variances for plots
            var += (t_var + t_mean ** 2) * effective_weight[idx] / normalizer
    var -= mean ** 2
    std = np.sqrt(var)

    return mean, var, std


# Single-model predictions after EM
# assume single input and output
def single_prediction(dpglm, data):
    nb_data = len(data)

    inputs = data[:, :1]
    outputs = data[:, 1:]
    prediction = np.zeros((nb_data, 1))

    for i in range(nb_data):
        idx = dpglm.labels_list[0].z[i]
        prediction[i, :] = dpglm.components[idx].predict(inputs[i, :])

    import matplotlib.pyplot as plt
    plt.scatter(inputs, outputs[:, 0], s=1)
    plt.scatter(inputs, prediction[:, 0], color='red', s=1)
    plt.show()


# Sample single-model predictions from posterior
# assume single input and output
def sample_prediction(dpglm, data, n_draws=25):
    nb_data = len(data)

    inputs = data[:, :1]
    outputs = data[:, 1:]
    prediction_samples = np.zeros((n_draws, nb_data, 1))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(25, 12))
    for d in range(n_draws):
        dpglm.resample_model()
        for i in range(nb_data):
            idx = dpglm.labels_list[0].z[i]
            prediction_samples[d, i, :] = dpglm.components[idx].predict(inputs[i, :])

        ax = fig.add_subplot(5, 5, d + 1)
        ax.scatter(inputs, outputs, s=1)
        ax.scatter(inputs, prediction_samples[d, ...], color='red', s=1)
    plt.show()

    prediction_mean = np.mean(prediction_samples, axis=0)
    plt.figure(figsize=(16, 6))
    plt.scatter(inputs, outputs, s=1)
    plt.scatter(inputs, prediction_mean, color='red', s=1)
    plt.show()
