import numpy as np
from scipy.stats import t
from mimo.util.general import inv_psd
from scipy.special import gamma, multigammaln
from mimo import distributions

def predict_train(dpglm, data, out_dim, in_dim_niw):
    pred_y = np.zeros((data.shape[0], out_dim))
    err_squared = 0
    err = 0
    for i in range(data.shape[0]):
        idx = dpglm.labels_list[0].z[i]
        x = data[i, :-out_dim]
        y = data[i, in_dim_niw:]
        pred_y[i] = dpglm.components[idx].predict(x)
        err = err + np.absolute((y - pred_y[i]))
        err_squared = err_squared + ((y - pred_y[i]) ** 2)
    return pred_y, err, err_squared

def get_component_standard_parameters(component):
    mu = component.gaussian.mu
    kappa = component.kappa
    psi_niw = component.invwishart_niw.psi
    nu_niw = component.invwishart_niw.nu
    M = component.matnorm.M
    V = component.matnorm.V,
    psi_mniw = component.invwishart_mniw.psi
    nu_mniw = component.invwishart_mniw.nu
    return mu, kappa, psi_niw, nu_niw, M, V, psi_mniw, nu_mniw

def student_t(x_hat, m_k, beta_k, W_k, v_k, D):
    L_k = ((v_k + 1 - D) * beta_k) / (1 + beta_k) * (W_k**-1)
    df = v_k + 1 + D
    L_k = L_k ** -1
    prob = t.pdf(x_hat, df, m_k, L_k)
    return prob


def matrix_t(data, idx, labels, out_dim, in_dim_niw, affine, x_hat, V_k, M_k, nb_models, S_0, dot_xx, dot_yx, dot_yy, psi_mniw, nu_mniw, N_0):
    def inv(arg):
        if isinstance(arg,int):
            arg = 1 / arg
        else:
            arg = np.linalg.inv(arg)
        return arg
    def transp(arg):
        if isinstance(arg,int):
            arg = arg
        else:
            arg = arg.T
        return arg

    # x_hat = transp(x_hat)   # correct shape of x

    if affine:
        b = np.ones((1, len(x_hat)+ 1))
        b[:, :-1] = x_hat
        # x_hat = np.asarray([[x_hat[0], 1]])
        x_hat = b
    x_hat = x_hat.T  # correct shape of x

    # bincount = np.zeros(nb_models)
    # for i in range(nb_models):
    #     bincount[i] = int(np.count_nonzero(labels == i))
    # if int(bincount[idx]) == 0:
    #     return np.asarray(0)
    #
    # if affine:
    #     x = np.zeros([in_dim_niw+1,int(bincount[idx])])
    # else:
    #     x = np.zeros([in_dim_niw,int(bincount[idx])])
    # y = np.zeros([out_dim,int(bincount[idx])])
    #
    # iter = 0
    # for i in range(len(labels)):
    # # for i in range(len(data[:, 0])):
    # # x = np.empty_like(data[:, :]).T
    #
    #     if labels[i] == idx:
    #         x[:, iter] = data[i, :-out_dim]
    #         if affine:
    #             x[1, iter] = 1
    #         y[:, iter] = data[i, in_dim_niw:]
    #         iter = iter + 1#
    # dot_xx = np.dot(x, x.T)
    # dot_yx = np.dot(y, x.T)
    # dot_yy = np.dot(y, y.T)

    dot_x_hat_x_hat = np.dot(x_hat, x_hat.T)

    S_xx = dot_xx + V_k[0]
    S_yx = dot_yx + np.dot(M_k, V_k[0])
    S_yy = dot_yy + np.dot(M_k, np.dot(V_k, M_k.T))
    S_y_x = S_yy - np.dot(S_yx, np.dot(inv(S_xx), S_yx.T))
    term = inv(S_xx + dot_x_hat_x_hat )

    c = 1 - np.dot(transp(x_hat),np.dot(term,x_hat))

    mean_matrix_T = np.dot(S_yx, np.dot(inv(S_xx),x_hat))

    if out_dim  == 1:
        std_matrix_T = np.sqrt((S_y_x + S_0) * ( 1 / c) / nu_mniw + N_0 + 1 - 2)  # variance of a matrix-T is scale-parameter divided by df -2 (see wikipedia and minka - bayesian linear regression)
    else:
        std_matrix_T = 0 #Fixme to multivariate output

    # mean_matrix_T = np.dot(M_k,x_hat)
    # std_matrix_T = np.sqrt(psi_mniw / nu_mniw-out_dim-1 )

    return mean_matrix_T, std_matrix_T


# see murphy (2007) - conjugate bayesian analysis of the gaussian distribution, marginal likelihood for a matrix-inverse-wishart prior
def NIW_marg_likelihood(data_x, mu, kappa, psi_niw, nu_niw, n, D):
    def det(arg):
        arg = np.absolute(arg)
        if int(D) == 1:
            arg = np.absolute(arg)
        else:
            arg = np.linalg.det(arg)
        return arg

    data_x = np.asarray([data_x])

    hypparams = dict(mu=mu, kappa=kappa, psi=psi_niw, nu=nu_niw)
    prior = distributions.NormalInverseWishart(**hypparams)
    model = distributions.BayesianGaussian(prior=prior)

    log_partition_prior = model.prior.log_partition()

    model.meanfieldupdate(data_x)

    log_term1 = np.log((1 / 2* np.pi) ** (n*D / 2))
    log_partition_posterior = model.posterior.log_partition()

    log_marg_likelihood = log_term1 + log_partition_posterior - log_partition_prior
    marg_likelihood = np.exp(log_marg_likelihood)

    # kappa_hat = model.posterior.kappa
    # psi_niw_hat = model.posterior.invwishart.psi
    # nu_niw_hat = model.posterior.invwishart.nu

    # # term1 =  (1 / np.pi) ** (n*D / 2)
    # log_term1 = np.log((1 / np.pi) ** (n*D / 2))
    # term1 = np.exp(log_term1)
    #
    # log_term2 = multigammaln(nu_niw_hat / 2, D) - multigammaln(nu_niw / 2, D)
    # term2 = np.exp(log_term2)
    #
    # log_term3 = nu_niw / 2 * np.log(det(psi_niw)) - nu_niw_hat / 2 * np.log(det(psi_niw_hat))
    # term3 = np.exp(log_term3)
    #
    # log_term4 = np.log((kappa / kappa_hat) ** (D/2))
    # term4 = (kappa / kappa_hat) ** (D/2)
    #
    # if np.isnan(term1):
    #     term1 = 1e-16
    #
    # if np.isnan(term2):
    #     term2 = 1e-16
    #
    # if np.isnan(term3):
    #     term3 = 1e-16
    # if np.isinf(term3):
    #     term3 = 1e+5
    #
    # if np.isnan(term4):
    #     term4 = 1e-16
    #
    # marg_likelihood = term1 * term2 * term3 * term4
    # # marg_likelihood = marg_likelihood / 1e-10
    # if np.isnan(marg_likelihood):
    #     marg_likelihood = 1e-16
    # if np.isinf(marg_likelihood):
    #     marg_likelihood = 1e+5
    # log_marg_likelihood = log_term1 * log_term2 * log_term3 * log_term4
    # log_marg_likelihood = log_term1 + log_term2 + log_term3 + log_term4
    # marg_likelihood = np.exp(log_marg_likelihood)

    return marg_likelihood
