import numpy as np

from mimo import distributions


# assume single input and output
def sample_prediction(dpglm, data, n_draws=25):
    n_train = data.shape[0]
    _train_inputs = data[:, :1]
    _train_outputs = data[:, 1:]
    _prediction_samples = np.zeros((n_draws, n_train, 1))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(25, 12))
    for d in range(n_draws):
        dpglm.resample_model()
        for i in range(n_train):
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


def matrix_t(query, prior, posterior, stats):
    mu, kappa, psi_niw, nu_niw, Mk, Vk, psi_mniw, nu_mniw = posterior.params
    _, _, _, _, _, _, S0, N0 = prior.params

    xxT, yxT, yyT = stats['xxT'], stats['yxT'], stats['yyT']

    q = query
    if prior.affine:
        q = np.hstack((query, 1.))

    qqT = np.outer(q, q)

    Sxx = xxT + Vk[0]
    Syx = yxT + np.dot(Mk, Vk[0])
    Syy = yyT + np.dot(Mk, np.dot(Vk, Mk.T))
    _Syx = Syy - np.dot(Syx, np.dot(np.linalg.inv(Sxx), Syx.T))
    c = 1. - q.T @ np.linalg.inv(Sxx + qqT) @ q

    df = nu_mniw + N0 + 1
    mean = np.dot(Syx, np.dot(np.linalg.inv(Sxx), q))

    # Variance of a matrix-T is scale-parameter divided by (df - 2)
    # (see Wikipedia and Minka - Bayesian linear regression)
    var = 1. / c * (_Syx + S0) / (df - 2)

    return mean, var, df


# see Murphy (2007) - Conjugate bayesian analysis of the gaussian distribution,
# marginal likelihood for a matrix-inverse-wishart prior
def niw_marginal_likelihood(data, prior):
    # Note the passed prior is actually the inferred posterior
    # copy parameters of the input Normal-Inverse-Wishart posterior
    mu, kappa = prior.gaussian.mu, prior.kappa
    psi, nu = prior.invwishart_niw.psi, prior.invwishart_niw.nu

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
