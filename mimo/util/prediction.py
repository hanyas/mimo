import numpy as np
from mimo import distributions


# Marginalize prediction under posterior
def meanfield_prediction(dpglm, data, input_dim, output_dim, prior='stick-breaking'):
    nb_data = len(data)
    nb_models = len(dpglm.components)

    # initialize variables
    mean, var, = np.zeros((nb_data, output_dim)), np.zeros((nb_data, output_dim)),

    # compute posterior mixing weights
    weights = None
    if prior == 'dirichlet':
        weights = dpglm.gating.posterior.alphas
    elif prior == 'stick-breaking':
        product = np.ones((nb_models, ))
        gammas = dpglm.gating.posterior.gammas
        deltas = dpglm.gating.posterior.deltas
        for idx, c in enumerate(dpglm.components):
            product[idx] = gammas[idx] / (gammas[idx] + deltas[idx])
            for j in range(idx):
                product[idx] = product[idx] * (1. - gammas[j] / (gammas[j] + deltas[j]))
        weights = product

    # prediction / mean function of yhat for all test data xhat
    for i in range(nb_data):
        xhat = data[i, :input_dim]

        # calculate the marginal likelihood of test data xhat for each cluster
        mlklhd = np.zeros((nb_models,))
        # calculate the normalization term for mean function for xhat
        normalizer = 0.
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                mlklhd[idx] = niw_marginal_likelihood(xhat, c.posterior)
                normalizer = normalizer + weights[idx] * mlklhd[idx]

        # calculate contribution of each cluster to mean function
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                t_mean, t_var, _ = matrix_t(xhat, c.posterior)
                t_var = np.diag(t_var)  # consider only diagonal variances for plots

                # Mean of a mixture = sum of weighted means
                mean[i, :] += t_mean * mlklhd[idx] * weights[idx] / normalizer

                # Variance of a mixture = sum of weighted variances + ...
                # ... + sum of weighted squared means - squared sum of weighted means
                var[i, :] += (t_var + t_mean ** 2) * mlklhd[idx] * weights[idx] / normalizer
        var[i, :] -= mean[i, :] ** 2

    return mean, var


# Weighted EM predictions over all models
def em_prediction(dpglm, data, input_dim, output_dim):
    nb_data = len(data)
    nb_models = len(dpglm.components)

    # initialize variables
    mean, var, = np.zeros((nb_data, output_dim)), np.zeros((nb_data, output_dim)),

    # mixing weights
    weights = dpglm.gating.probs

    # prediction / mean function of yhat for all test data xhat
    for i in range(nb_data):
        xhat = data[i, :input_dim]

        # calculate the marginal likelihood of test data xhat for each cluster
        lklhd = np.zeros((nb_models,))
        # calculate the normalization term for mean function for xhat
        normalizer = 0.
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                lklhd[idx] = gaussian_likelihood(xhat, c)
                normalizer = normalizer + weights[idx] * lklhd[idx]

        # calculate contribution of each cluster to mean function
        for idx, c in enumerate(dpglm.components):
            if idx in dpglm.used_labels:
                t_mean = c.predict(xhat)
                t_var = np.diag(c.sigma)  # consider only diagonal variances for plots

                # Mean of a mixture = sum of weighted means
                mean[i, :] += t_mean * lklhd[idx] * weights[idx] / normalizer

                # Variance of a mixture = sum of weighted variances + ...
                # ... + sum of weighted squared means - squared sum of weighted means
                var[i, :] += (t_var + t_mean ** 2) * lklhd[idx] * weights[idx] / normalizer
        var[i, :] -= mean[i, :] ** 2

    return mean, var


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
            for idx, c in enumerate(dpglm.components):
                if idx in dpglm.used_labels:

                    t_mean = c.predict(xhat)
                    t_var = np.diag(c.sigma) # consider only diagonal variances for plots

                    # Mean of mixture = sum of weighted, sampled means
                    mean[i, :] += t_mean * lklhd[idx] * weights[idx] / (gibbs_samples * normalizer)

                    # Variance of a mixture = sum of weighted variances + ...
                    # ... + sum of weighted squared means - squared sum of weighted means
                    var[i, :] += t_var * lklhd[idx] * weights[idx] / (gibbs_samples * normalizer)
                    # var[i, :] += (t_var + t_mean ** 2) * lklhd[idx] * weights[idx] / (gibbs_samples * normalizer)
            # var[i, :] -= mean[i, :] ** 2 / gibbs_samples
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
                t_mean = component.predict(xhat)
                t_var = np.diag(component.sigma)  # consider only diagonal variances for plots

                # Mean of mixture = sum of weighted, sampled means
                mean[i, :] += t_mean * lklhd[j] / (gibbs_samples * normalizer)

                # Variance of a mixture = sum of weighted variances + ...
                # ... + sum of weighted squared means - squared sum of weighted means
                var[i, :] += t_var * lklhd[j] / (gibbs_samples * normalizer)
                # var[i, :] += (t_var + t_mean ** 2) * lklhd[j] / (gibbs_samples * normalizer)
        # var[i, :] -= mean[i, :] ** 2 / gibbs_samples
    return mean, var

# Single-model predictions after EM
# assume single input and output
def single_prediction(dpglm, data):
    nb_data = len(data)

    _train_inputs = data[:, :1]
    _train_outputs = data[:, 1:]
    _prediction = np.zeros((nb_data, 1))

    for i in range(nb_data):
        idx = dpglm.labels_list[0].z[i]
        _prediction[i, :] = dpglm.components[idx].predict(_train_inputs[i, :])

    import matplotlib.pyplot as plt
    plt.scatter(_train_inputs, _train_outputs, s=1)
    plt.scatter(_train_inputs, _prediction, color='red', s=1)
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


def matrix_t(query, posterior):
    mu, kappa, psi_niw, nu_niw, Mk, Vk, psi_mniw, nu_mniw = posterior.params

    q = query
    if posterior.affine:
        q = np.hstack((query, 1.))

    qqT = np.outer(q, q)

    df = nu_mniw + 1
    c = 1. - q.T @ np.linalg.inv(np.linalg.inv(Vk) + qqT) @ q
    mean = Mk @ q
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
    const = np.log(1. / (2. * np.pi) ** (1 * prior.dim / 2))

    log_marginal_likelihood = const + log_partition_posterior - log_partition_prior
    marginal_likelihood = np.exp(log_marginal_likelihood)

    return marginal_likelihood
