import numpy as np


def model_dimension(model):
    """ Compute the number of free parameters of a SPCA model.
    Cf. Proposition 2 for the formula. We remove the shift parameter and consider
    SPCA as a covariance model, as explained in Subsection 4.1.
    """
    d = len(model)
    p = np.sum(model)
    dim = p * (p - 1) / 2 - np.sum(np.array(model) * (np.array(model) - 1) / 2)
    return int(d + dim)


def evd(X):
    """ Perform the eigenvalue decomposition (EVD) of the sample covariance matrix of X.
    """
    n, p = X.shape
    mu = np.mean(X, axis=0)
    S = 1 / n * ((X - mu).T @ (X - mu))
    eigval, eigvec = np.linalg.eigh(S)
    eigval, eigvec = np.flip(eigval, -1), np.flip(eigvec, -1)
    eigval = np.clip(eigval, 0, np.inf)
    return eigval, eigvec, mu, n, p


def maximum_log_likelihood(X, model, return_params=False):
    """ Compute the maximum log likelihood of an SPCA model for a dataset X. Cf. Equation (11).
    """
    eigval, eigvec, mu_ML, n, p = evd(X)
    return maximum_log_likelihood_fast(eigval, eigvec, mu_ML, n, p, model, return_params)


def maximum_log_likelihood_fast(eigval, eigvec, mu_ML, n, p, model, return_params=False):
    """ A version of maximum_log_likelihood where we have already done the EVD.
    Used to speed up model selection, where the same evd is done for all models.
    """
    signature_0 = (0,) + tuple(np.cumsum(model))
    eigval_ML = np.concatenate([[np.mean(eigval[dk_1:dk])] * model for (dk_1, dk, model) in zip(signature_0[:-1], signature_0[1:], model)])
    ML = - (n / 2) * (p * np.log(2 * np.pi) + np.sum(np.log(eigval_ML)) + p)
    if return_params:
        Sigma_ML = eigvec @ np.diag(eigval_ML) @ eigvec.T
        return ML, mu_ML, Sigma_ML
    else:
        return ML


def bic(X, model):
    """ Compute the Bayesian Information Criterion (BIC) of an SPCA model for a dataset X. Cf. Equation (14).
    """
    n, p = X.shape
    return 1 / n * (model_dimension(model) * np.log(n) - 2 * maximum_log_likelihood(X, model))


def bic_fast(eigval, eigvec, mu_ML, n, p, model):
    """ A version of bic where we have already done the EVD.
    Used to speed up model selection, where the same evd is done for all models.
    """
    return 1 / n * (model_dimension(model) * np.log(n) - 2 * maximum_log_likelihood_fast(eigval, eigvec, mu_ML, n, p, model))