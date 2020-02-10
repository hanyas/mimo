import numpy as np
from mimo import distributions

from mimo.distributions.dirichlet import Dirichlet
from mimo.distributions.dirichlet import StickBreaking

from pathos.multiprocessing import ProcessingPool as Pool


def meanfield_forcast(dpglm, query, horizon=1,
                      exogenous=None, incremental=True):

    if exogenous is not None:
        assert horizon <= len(exogenous)

    _output = query
    output = [query]
    for h in range(horizon):
        if exogenous is not None:
            _input = np.hstack((output[-1], exogenous[h, :]))
        else:
            _input = _output

        _prediction, _, _ = meanfield_prediction(dpglm, _input)
        if incremental:
            _output = _output + _prediction
        else:
            _output = _prediction

        output.append(_output)

    return np.vstack(output)


def scaled_meanfield_forcast(dpglm, query, horizon=1,
                             exogenous=None, incremental=True,
                             input_scaler=None, target_scaler=None):

    if exogenous is not None:
        assert horizon <= len(exogenous)

    _output = query
    output = [query]
    for h in range(horizon):
        if exogenous is not None:
            _input = np.hstack((output[-1], exogenous[h, :]))
        else:
            _input = _output

        _scaled_input = np.squeeze(input_scaler.transform(np.atleast_2d(_input)))
        _scaled_prediction, _, _ = meanfield_prediction(dpglm, _scaled_input)
        if incremental:
            _unscaled_prediction = target_scaler.inverse_transform(np.atleast_2d(_scaled_prediction))
            _output = _output + np.squeeze(_unscaled_prediction)
        else:
            _unscaled_prediction = target_scaler.inverse_transform(np.atleast_2d(_scaled_prediction))
            _output = np.squeeze(_unscaled_prediction)

        output.append(_output)

    return np.vstack(output)


def parallel_meanfield_prediction(dpglm, query, prediction,
                                  input_scaler=None, target_scaler=None):
    query = np.atleast_2d(query)

    nb_data = len(query)
    nb_dim = dpglm.components[0].dout

    def _loop(n):
        if input_scaler is None or target_scaler is None:
            return meanfield_prediction(dpglm, query[n, :], prediction)
        else:
            return scaled_meanfield_prediction(dpglm, query[n, :], prediction,
                                               input_scaler, target_scaler)

    _pool = Pool(processes=-1)
    res = _pool.map(_loop, range(nb_data))

    res = np.asarray(res).squeeze()
    mean, var, std = res[:, :nb_dim], res[:, nb_dim:2 * nb_dim], res[:, 2 * nb_dim:]

    return mean, var, std


def scaled_meanfield_prediction(dpglm, query, prediction='average',
                                input_scaler=None, target_scaler=None):

    # scale query
    scaled_query = input_scaler.transform(np.atleast_2d(query)).squeeze()

    # predict in scaled space
    scaled_mean, scaled_var, _ = meanfield_prediction(dpglm, scaled_query, prediction)

    # unscale mean and var
    mean = target_scaler.inverse_transform(np.atleast_2d(scaled_mean))
    trans = np.sqrt(target_scaler.explained_variance_[:, None]) * target_scaler.components_
    var = trans.T @ scaled_var @ trans

    return mean, var, np.sqrt(var)


# Marginalize prediction under posterior
def meanfield_prediction(dpglm, query, prediction='average'):
    nb_models = len(dpglm.components)
    nb_dim = dpglm.components[0].dout

    # initialize variables
    mean, var, std = np.zeros((nb_dim, )), np.zeros((nb_dim, )), np.zeros((nb_dim, ))

    # compute posterior mixing weights
    weights = None
    if isinstance(dpglm.gating.prior, Dirichlet):
        weights = dpglm.gating.posterior.alphas
    elif isinstance(dpglm.gating.prior, StickBreaking):
        weights = stick_breaking(dpglm.gating.posterior.gammas,
                                 dpglm.gating.posterior.deltas)

    # calculate the marginal likelihood of query for each cluster
    # calculate the normalization term for mean function for xhat
    normalizer = 0.
    marginal_likelihood = np.zeros((nb_models,))
    effective_weight = np.zeros((nb_models,))
    for idx, c in enumerate(dpglm.components):
        if idx in dpglm.used_labels:
            marginal_likelihood[idx] = niw_marginal_likelihood(query, c.posterior)
            effective_weight[idx] = weights[idx] * marginal_likelihood[idx]
            normalizer = normalizer + weights[idx] * marginal_likelihood[idx]

    if prediction == 'mode':
        mode = np.argmax(effective_weight)
        t_mean, t_var, _ = predictive_matrix_t(query, dpglm.components[mode].posterior)
        t_var = np.diag(t_var)  # consider only diagonal variances for plots

        mean, var, std = t_mean, t_var, np.sqrt(t_var)

    elif prediction == 'average':
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                t_mean, t_var, _ = predictive_matrix_t(query, c.posterior)
                t_var = np.diag(t_var)  # consider only diagonal variances for plots

                # Mean of a mixture = sum of weighted means
                mean += t_mean * effective_weight[idx] / normalizer

                # Variance of a mixture = sum of weighted variances + ...
                # ... + sum of weighted squared means - squared sum of weighted means
                # var[i, :] += (t_var) * mlkhd[idx] * weights[idx] / normalizer
                var += (t_var + t_mean ** 2) * effective_weight[idx] / normalizer
        var -= mean ** 2
        std = np.sqrt(var)

    else:
        raise NotImplementedError

    return mean, var, std


# Weighted EM predictions over all models
def em_prediction(dpglm, query):
    nb_models = len(dpglm.components)
    nb_dim = dpglm.components[0].dout

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
            lklhd[idx] = gaussian_likelihood(query, c)
            effective_weight[idx] = weights[idx] * lklhd[idx]
            normalizer = normalizer + weights[idx] * lklhd[idx]

    # calculate contribution of each cluster to mean function
    for idx, c in enumerate(dpglm.components):
        if idx in dpglm.used_labels:
            t_mean = c.predict(query)
            t_var = np.diag(c.sigma)  # consider only diagonal variances for plots

            # Mean of a mixture = sum of weighted means
            mean += t_mean * effective_weight[idx] / normalizer

            # Variance of a mixture = sum of weighted variances + ...
            # ... + sum of weighted squared means - squared sum of weighted means
            var += (t_var + t_mean ** 2) * effective_weight[idx] / normalizer
    var -= mean ** 2
    std = np.sqrt(var)

    return mean, var, std


# Prediction using posterior samples of the Gibbs sampler;
# weighted by likelihood of test data and cluster weights, averaged over gibbs iterations
def gibbs_prediction(dpglm, test_data, train_data, input_dim, output_dim, gibbs_samples, gating_prior, affine):
    nb_data_test = len(test_data)
    nb_models = len(dpglm.components)

    # initialize variables
    mean, var, = np.zeros((nb_data_test, output_dim)), np.zeros((nb_data_test, output_dim)),

    # mixing weights
    weights = dpglm.gating.probs

    # average over samples
    for g in range(gibbs_samples):

        # de-correlate samples / subsample markov chain; no subsampling for gibbs_step == 1
        gibbs_step = 1
        for r in range(gibbs_step):
            dpglm.resample_model()

        # calculate marginal likelihood of all test data xhat
        # assuming identical prior parameters for each
        niw_marginal_normalized = np.zeros((nb_data_test, output_dim))
        for i in range(nb_data_test):
            xhat = test_data[i, :input_dim]
            niw_marginal_normalized[i] = niw_marginal_likelihood(xhat, dpglm.components[0].prior)
        niw_marginal_normalized = niw_marginal_normalized / niw_marginal_normalized.sum()

        # prediction / mean function of yhat for all test data xhat
        for i in range(nb_data_test):
            xhat = test_data[i, :input_dim]

            # calculate the likelihood of test data xhat for each cluster
            # calculate the normalization term for mean function for xhat
            lklhd = np.zeros((nb_models,))
            normalizer = 0.
            for idx, c in enumerate(dpglm.components):
                if idx in dpglm.used_labels:
                    lklhd[idx] = gaussian_likelihood(xhat, c)
                    normalizer = normalizer + weights[idx] * lklhd[idx]

            # # if stick-breaking, consider prediction of a new cluster with probability proportional to concentration parameter
            # # assuming identical hyperparameters for all components
            # if gating_prior == 'stick-breaking':
            #
            #     # get prior parameter
            #     delta = dpglm.gating.prior.deltas[0]
            #     Mk = dpglm.components[0].prior.matnorm.M
            #
            #     # add term to normalizer
            #     # normalizer = normalizer + delta * niw_marginal_likelihood(xhat, dpglm.components[0].prior)
            #     normalizer = normalizer + delta * niw_marginal_normalized[i]
            #
            #     # prediction for prior parameters
            #     if affine:
            #         Mk, b = Mk[:, :-1], Mk[:, -1]
            #         y = xhat.dot(Mk.T) + b.T
            #     else:
            #         y = xhat.dot(Mk.T)
            #
            #     # add term to mean function
            #     mean[i, :] += delta * y * niw_marginal_normalized[i] / (gibbs_samples * normalizer)

            # calculate contribution of each existing cluster to mean function
            for idx, c in enumerate(dpglm.components):
                if idx in dpglm.used_labels:

                    Mk = c.posterior.matnorm.M
                    # prediction for prior parameters
                    if affine:
                        Mk, b = Mk[:, :-1], Mk[:, -1]
                        y = xhat.dot(Mk.T) + b.T
                    else:
                        y = xhat.dot(Mk.T)

                    # t_mean = c.predict(xhat)
                    t_mean = y
                    t_var = np.diag(c.sigma)  # consider only diagonal variances for plots

                    # Mean of mixture = sum of weighted, sampled means
                    mean[i, :] += t_mean * lklhd[idx] * weights[idx] / (gibbs_samples * normalizer)

                    # Variance of a mixture = sum of weighted variances + ...
                    # ... + sum of weighted squared means - squared sum of weighted means
                    var[i, :] += t_var * lklhd[idx] * weights[idx] / (
                            gibbs_samples * normalizer)  # var[i, :] += (t_var + t_mean ** 2) * lklhd[idx] * weights[idx] / (gibbs_samples * normalizer)  # var[i, :] -= mean[i, :] ** 2 / gibbs_samples
    return mean, var


# Prediction using posterior samples of the Gibbs sampler
# weighted by test data likelihood; not weighted by cluster weights; averaged over gibbs iterations
def gibbs_prediction_noWeights(dpglm, test_data, train_data, input_dim, output_dim, gibbs_samples, gating_prior, affine):
    nb_data_test = len(test_data)
    nb_data_train = len(train_data)

    # initialize variables
    mean, var, = np.zeros((nb_data_test, output_dim)), np.zeros((nb_data_test, output_dim)),

    # average over samples
    for g in range(gibbs_samples):

        # de-correlate samples / subsample markov chain; no subsampling for gibbs_step == 1
        gibbs_step = 1
        for r in range(gibbs_step):
            dpglm.resample_model()

        # calculate marginal likelihood of all test data xhat
        # assuming identical prior parameters for each component
        niw_marginal_normalized = np.zeros((nb_data_test, output_dim))
        for i in range(nb_data_test):
            xhat = test_data[i, :input_dim]
            niw_marginal_normalized[i] = niw_marginal_likelihood(xhat, dpglm.components[0].prior)
        niw_marginal_normalized = niw_marginal_normalized / niw_marginal_normalized.sum()

        # prediction / mean function of yhat for all test data xhat
        for i in range(nb_data_test):
            xhat = test_data[i, :input_dim]

            # calculate the marginal likelihood of test data xhat for each cluster
            # calculate the normalization term for mean function for xhat
            lklhd = np.zeros((nb_data_train,))
            normalizer = 0.
            for j in range(nb_data_train):
                idx = dpglm.labels_list[0].z[j]
                component = dpglm.components[idx]
                lklhd[j] = gaussian_likelihood(xhat, component)
                normalizer = normalizer + lklhd[j]

            # if stick-breaking, consider prediction of a new cluster with probability proportional to concentration parameter
            # assuming identical hyperparameters for all components
            if gating_prior == 'stick-breaking':

                # get prior parameter
                delta = dpglm.gating.prior.deltas[0]
                Mk = dpglm.components[0].prior.matnorm.M

                # add term to normalizer
                # normalizer = normalizer + delta * niw_marginal_likelihood(xhat, dpglm.components[0].prior)
                normalizer = normalizer + delta * niw_marginal_normalized[i]

                # prediction for prior parameters
                if affine:
                    Mk, b = Mk[:, :-1], Mk[:, -1]
                    y = xhat.dot(Mk.T) + b.T
                else:
                    y = xhat.dot(Mk.T)

                # add term to mean function
                mean[i, :] += delta * y * niw_marginal_normalized[i] / (gibbs_samples * normalizer)

            # calculate contribution of each existing cluster to mean function
            for j in range(nb_data_train):
                idx = dpglm.labels_list[0].z[j]
                component = dpglm.components[idx]

                # calculate contribution of each cluster to mean function
                t_mean = component.predict(xhat)  # Fixme
                t_var = np.diag(component.sigma)  # consider only diagonal variances for plots

                # Mean of mixture = sum of weighted, sampled means
                mean[i, :] += t_mean * lklhd[j] / (gibbs_samples * normalizer)

                # Variance of a mixture = sum of weighted variances + ...
                # ... + sum of weighted squared means - squared sum of weighted means
                var[i, :] += t_var * lklhd[j] / (gibbs_samples * normalizer)  # var[i, :] += (t_var + t_mean ** 2) * lklhd[j] / (gibbs_samples * normalizer)  # var[i, :] -= mean[i, :] ** 2 / gibbs_samples
    return mean, var


# Single-model predictions after EM
# assume single input and output
def single_prediction(dpglm, data, input_dim, output_dim):
    nb_data = len(data)

    _train_inputs = data[:, :1]
    _train_outputs = data[:, input_dim:]
    _prediction = np.zeros((nb_data, output_dim))

    for i in range(nb_data):
        idx = dpglm.labels_list[0].z[i]
        _prediction[i, :] = dpglm.components[idx].predict(_train_inputs[i, :])

    import matplotlib.pyplot as plt
    plt.scatter(_train_inputs, _train_outputs[:, 0], s=1)
    plt.scatter(_train_inputs, _prediction[:, 0], color='red', s=1)
    plt.show()


# Sample single-model predictions from posterior
# assume single input and output
def sample_prediction(dpglm, data, n_draws=25):
    nb_data = len(data)

    _train_inputs = data[:, :1]
    _train_outputs = data[:, 1:]
    _prediction_samples = np.zeros((n_draws, nb_data, 1))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(25, 12))
    for d in range(n_draws):
        dpglm.resample_model()
        for i in range(nb_data):
            idx = dpglm.labels_list[0].z[i]
            _prediction_samples[d, i, :] = dpglm.components[idx].predict(_train_inputs[i, :])

        ax = fig.add_subplot(5, 5, d + 1)
        ax.scatter(_train_inputs, _train_outputs, s=1)
        ax.scatter(_train_inputs, _prediction_samples[d, ...], color='red', s=1)
    plt.show()

    _prediction_mean = np.mean(_prediction_samples, axis=0)
    plt.figure(figsize=(16, 6))
    plt.scatter(_train_inputs, _train_outputs, s=1)
    plt.scatter(_train_inputs, _prediction_mean, color='red', s=1)
    plt.show()


def gaussian_likelihood(data, distribution):
    mu, sigma = distribution.mu, distribution.sigma_niw
    model = distributions.Gaussian(mu=mu, sigma=sigma)

    log_likelihood = model.log_likelihood(data)
    log_partition = model.log_partition()

    likelihood = np.exp(log_partition + log_likelihood)
    return likelihood


def predictive_matrix_t(query, posterior):
    mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw = posterior.params

    q = query
    if posterior.affine:
        q = np.hstack((query, 1.))

    qqT = np.outer(q, q)

    df = nu_mniw + 1
    mean = M @ q
    c = 1. - q.T @ np.linalg.inv(np.linalg.inv(V) + qqT) @ q
    var = 1. / c * psi_mniw / (df - 2)

    return mean, var, df


# see Murphy (2007) - Conjugate bayesian analysis of the gaussian distribution,
# marginal likelihood for a matrix-inverse-wishart prior
def niw_marginal_likelihood(data, posterior):
    # copy parameters of the input Normal-Inverse-Wishart posterior
    mu, kappa = posterior.gaussian.mu, posterior.kappa
    psi, nu = posterior.invwishart_niw.psi, posterior.invwishart_niw.nu

    hypparams = dict(mu=mu, kappa=kappa, psi=psi, nu=nu)
    prior = distributions.NormalInverseWishart(**hypparams)

    model = distributions.BayesianGaussian(prior=prior)
    model.meanfield_update(np.atleast_2d(data))

    log_partition_prior = model.prior.log_partition()
    log_partition_posterior = model.posterior.log_partition()
    const = np.log(1. / (2. * np.pi) ** (1. * prior.dim / 2.))

    log_marginal_likelihood = const + log_partition_posterior - log_partition_prior
    marginal_likelihood = np.exp(log_marginal_likelihood)

    return marginal_likelihood


# recover weights of a stick-breaking process
def stick_breaking(gammas, deltas):
    nb_models = len(gammas)
    product = np.ones((nb_models,))
    for idx in range(nb_models):
        product[idx] = gammas[idx] / (gammas[idx] + deltas[idx])
        for j in range(idx):
            product[idx] = product[idx] * (1. - gammas[j] / (gammas[j] + deltas[j]))

    return product
