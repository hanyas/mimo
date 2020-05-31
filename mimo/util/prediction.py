import numpy as np

import scipy as sc
from scipy import stats
from scipy.stats import multivariate_normal as mvn

from mimo.util.general import matrix_linear_studentt
from mimo.util.general import matrix_linear_gaussian
from mimo.util.general import multivariate_studentt_loglik as mvt_logpdf

import pathos
from pathos.pools import _ProcessPool as Pool
nb_cores = pathos.multiprocessing.cpu_count()

import time


def kstep_error(dpglm, state, exogenous=None, horizon=1,
                prediction='average', incremental=True,
                input_scaler=None, target_scaler=None,
                type='gaussian', sparse=False):

    start = time.process_time()

    from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

    mse, smse, evar = [], [], []
    for _state, _exo in zip(state, exogenous):
        _state_list, _exo_list,  = [], []

        nb_steps = _state.shape[0] - horizon
        for t in range(nb_steps):
            _state_list.append(_state[t, :])
            _exo_list.append(_exo[t: t + horizon, :])

        _forcast = parallel_meanfield_predictive_forcast(dpglm, _state_list,
                                                         _exo_list, horizon,
                                                         prediction, incremental,
                                                         input_scaler, target_scaler,
                                                         type, sparse)

        target, forcast = [], []
        for t in range(nb_steps):
            target.append(_state[t + horizon, :])
            forcast.append(_forcast[t][-1, :])

        target = np.vstack(target)
        forcast = np.vstack(forcast)

        _mse = mean_squared_error(target, forcast)
        _smse = 1. - r2_score(target, forcast, multioutput='variance_weighted')
        _evar = explained_variance_score(target, forcast, multioutput='variance_weighted')

        mse.append(_mse)
        smse.append(_smse)
        evar.append(_evar)

    finish = time.process_time()
    return np.mean(mse), np.mean(smse), np.mean(evar), finish - start


def parallel_meanfield_predictive_forcast(dpglm, state, exogenous=None, horizon=1,
                                          prediction='average', incremental=True,
                                          input_scaler=None, target_scaler=None,
                                          type='gaussian', sparse=False):

    assert isinstance(state, list)

    def _loop(kwargs):
        return meanfield_predictive_forcast(dpglm, kwargs['state'],
                                            kwargs['exogenous'], horizon,
                                            prediction, incremental,
                                            input_scaler, target_scaler,
                                            type, sparse)

    nb_traj = len(state)

    kwargs_list = []
    for n in range(nb_traj):
        _state = state[n]
        _exo = None if exogenous is None else exogenous[n]
        kwargs = {'state': _state, 'exogenous': _exo}
        kwargs_list.append(kwargs)

    with Pool(processes=nb_cores) as p:
        res = p.map(_loop, kwargs_list, chunksize=int(nb_traj / nb_cores))

    return res


def meanfield_predictive_forcast(dpglm, state, exogenous=None, horizon=1,
                                 prediction='average', incremental=True,
                                 input_scaler=None, target_scaler=None,
                                 type='gaussian', sparse=False):

    if exogenous is not None:
        assert horizon <= len(exogenous)

    _state = state
    forcast = [_state]
    for h in range(horizon):
        _query = _state
        if exogenous is not None:
            _exo = exogenous[h, :]
            _query = np.hstack((_state, _exo))

        _state = meanfield_prediction(dpglm, _query, None,
                                      prediction, incremental,
                                      input_scaler, target_scaler,
                                      type, sparse)[0]
        forcast.append(_state)

    return np.vstack(forcast)


def parallel_meanfield_prediction(dpglm, query, target=None,
                                  prediction='average', incremental=False,
                                  input_scaler=None, target_scaler=None,
                                  type='gaussian', sparse=False):

    def _loop(kwargs):
        return meanfield_prediction(dpglm, kwargs['x'], kwargs['y'],
                                    prediction, incremental,
                                    input_scaler, target_scaler,
                                    type, sparse)

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


def meanfield_predictive_gating(dpglm, input, sparse=False):
    nb_models = len(dpglm.components)

    # compute posterior mixing weights
    weights = dpglm.gating.posterior.mean()

    # calculate the marginal likelihood of query for each cluster
    # calculate the normalization term for mean function for query
    marginal_likelihood = np.zeros((nb_models, ))
    effective_weights = np.zeros((nb_models, ))

    _labels = dpglm.used_labels if sparse else range(nb_models)
    for idx in _labels:
        marginal_likelihood[idx] = np.exp(dpglm.components[idx].log_posterior_predictive_gaussian(input))
        effective_weights[idx] = weights[idx] * marginal_likelihood[idx]

    effective_weights = effective_weights / np.sum(effective_weights)

    return effective_weights


def meanfield_predictive_component(component, query, type='gaussian'):
    affine = component.posterior.mniw.affine
    params = component.posterior.mniw.params

    if type == 'studentt':
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
                         type='gaussian', sparse=False):

    nb_dim = dpglm.components[0].drow

    nb_models = len(dpglm.components)

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

    weights = meanfield_predictive_gating(dpglm, input, sparse)

    if prediction == 'mode':
        mode = np.argmax(weights)
        mean, _sigma, df = meanfield_predictive_component(dpglm.components[mode], input, type)

        if type == 'studentt':
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
        _labels = dpglm.used_labels if sparse else range(nb_models)
        for idx in _labels:
            _mu, _sigma, _df = meanfield_predictive_component(dpglm.components[idx], input, type)

            if type == 'studentt':
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
