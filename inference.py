import numpy as np


def model_dimension(gamma):
    """ Compute the number of free parameters of a given SPCA model.
    Cf. Proposition 2 for the formula. We remove the shift parameter $\mu$ and consider
    SPCA as a covariance model, as explained in Subsection 4.1.
    """
    d = len(gamma)
    p = np.sum(gamma)
    dim = p * (p - 1) / 2 - np.sum(np.array(gamma) * (np.array(gamma) - 1) / 2)
    return int(d + dim)


def maximum_log_likelihood(X, gamma, return_params=False):
    """ Compute the maximum log likelihood of an SPCA model of type $\gamma$
    for a dataset X. Cf. Equation (11).
    """
    signature_0 = (0,) + tuple(np.cumsum(gamma))
    n, p = X.shape
    mu_ML = np.mean(X, axis=0)
    S = 1 / n * ((X - mu_ML).T @ (X - mu_ML))
    eigval, eigvec = np.linalg.eigh(S)
    eigval, eigvec = np.flip(eigval, -1), np.flip(eigvec, -1)
    eigval = np.clip(eigval, 0, np.inf)
    eigval_ML = np.concatenate([[np.mean(eigval[dk_1:dk])] * gamma_k for (dk_1, dk, gamma_k) in zip(signature_0[:-1], signature_0[1:], gamma)])
    ML = - (n / 2) * (p * np.log(2 * np.pi) + np.sum(np.log(eigval_ML)) + p)
    if return_params:
        Sigma_ML = eigvec @ np.diag(eigval_ML) @ eigvec.T
        return ML, mu_ML, Sigma_ML
    else:
        return ML


def bic(X, gamma):
    """ Compute the Bayesian Information Criterion (BIC) of an SPCA model of type
    $\gamma$ for a dataset X. Cf. Equation (14).
    """
    n, p = X.shape
    return 1 / n * (model_dimension(gamma) * np.log(n) - 2 * maximum_log_likelihood(X, gamma))