import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.cluster.hierarchy import dendrogram

from inference import bic_fast, evd, maximum_log_likelihood_fast


def generate_models(p, family="SPCA", cardinal=None):
    """ Generate a subfamily of SPCA models.
    Family='SPCA' -> the whole family of SPCA models for a given dimension p.
    Family='PPCA' -> the whole family of PPCA models for a given dimension p.
    Family='IPPCA' -> the whole family of IPPCA models for a given dimension p.
    If a cardinal is specified, we only keep the models whose type has a given cardinal.
    """
    models = []
    for model_len in range(1, p + 1):
        if family == "SPCA" and cardinal is None:
            max_model_value = p - model_len + 1
            candidate_models = itertools.product(*[list(np.arange(1, max_model_value + 1)) for _ in range(model_len)])
            for model in candidate_models:
                if np.sum(model) == p:
                    models.append(model)
        elif family == "SPCA" and isinstance(cardinal, int) and (1 <= cardinal <= p):
            signature_list = list(itertools.combinations(np.arange(1, p), cardinal-1))
            models = [tuple(np.diff((0,) + signature + (p,))) for signature in signature_list]
        elif family == "PPCA":
            models.append((1,) * (model_len - 1) + (p - (model_len - 1),))
        elif family == "IPPCA":
            if model_len!=p:
                models.append((model_len, p-model_len))
        else:
            raise(NotImplementedError(f"The family {family} has not been implemented yet."))
    return models


def hierarchical_model_selection(eigval, dist=None, return_linkage_matrix=False):
    """ Perform model selection within the hierarchical clustering heuristic.
    Cf. Subsubsection 4.4.1 and Algorithm 1. We also deal with singular matrices.
    """
    eigval = list(eigval)
    eigval.sort(reverse=True)
    p = len(eigval)
    p_max = p if np.min(eigval) > 1e-12 else list((np.array(eigval)>1e-12)).index(False)
    candidate_models = [tuple([1] * (p_max - 1) + [p - (p_max - 1)] * 1)]
    signature_0 = (0,) + tuple(np.cumsum(candidate_models[0]))
    eigval_strat = [np.mean(eigval[di_1:di]) for (di_1, di, ni) in zip(signature_0[:-1], signature_0[1:], candidate_models[0])]
    eigengaps_strat = [dist(l1, l2) for (l1, l2) in zip(eigval_strat[:-1], eigval_strat[1:])]
    linkage_matrix = np.zeros((p_max - 1, 4))
    clusters_indexes = list(np.arange(p_max))
    clusters_cardinals = [1] * (p_max - 1) + [p - (p_max - 1)] * 1

    for i in range(p_max - 1):
        argmin_eigengap = np.argmin(eigengaps_strat)
        linkage_matrix[i, 0] = clusters_indexes[argmin_eigengap]
        linkage_matrix[i, 1] = clusters_indexes[argmin_eigengap + 1]
        linkage_matrix[i, 2] = eigengaps_strat[argmin_eigengap]
        linkage_matrix[i, 3] = clusters_cardinals[argmin_eigengap] + clusters_cardinals[
            argmin_eigengap + 1]

        clusters_indexes[argmin_eigengap] = p_max + i
        del clusters_indexes[argmin_eigengap + 1]

        clusters_cardinals[argmin_eigengap] += clusters_cardinals[argmin_eigengap + 1]
        del clusters_cardinals[argmin_eigengap + 1]

        model = list(candidate_models[-1])
        model[argmin_eigengap] += model[argmin_eigengap + 1]
        del model[argmin_eigengap + 1]
        candidate_models.append(tuple(model))

        signature_0 = (0,) + tuple(np.cumsum(model))
        eigval_strat = [
            np.mean(eigval[di_1:di]) for (di_1, di, ni) in
            zip(signature_0[:-1], signature_0[1:], model)
        ]
        eigengaps_strat = [dist(l1, l2) for (l1, l2) in zip(eigval_strat[:-1], eigval_strat[1:])]
    if return_linkage_matrix:
        return candidate_models, linkage_matrix
    else:
        return candidate_models


def type_length_model_selection(X, cardinal, criterion="BIC"):
    """ Perform model selection within the fixed type length heuristic.
    Cf. Subsubsection 4.4.2.
    """
    eigval, eigvec, mu_ML, n, p = evd(X)
    criterion_fun = (lambda eigval, eigvec, mu_ML, n, p, model: - maximum_log_likelihood_fast(eigval, eigvec, mu_ML, n, p, model)) if criterion == "ML" else bic_fast
    models = generate_models(p, family="SPCA", cardinal=cardinal)
    best_model = models[0]
    best_score = criterion_fun(eigval, eigvec, mu_ML, n, p, best_model)
    for model in models[1:]:
        score = criterion_fun(eigval, eigvec, mu_ML, n, p, model)
        if score < best_score:
            best_model = model
            best_score = score
    return best_model, best_score


if __name__ == '__main__':
    np.random.seed(42)
    path = os.path.dirname(__file__) + "/figures/"

    p = 15
    n = 5000
    eigval_pop = [1]
    eigengaps_pop = np.random.rand(p) / 5
    eigengaps_pop[eigengaps_pop < .1] = 0
    for i in range(p - 1):
        eigval_pop.append(eigval_pop[-1] - eigengaps_pop[i])

    X = np.random.multivariate_normal(np.zeros(p), np.diag(eigval_pop), n)
    eigval, eigvec, mu, n, p = evd(X)

    models, linkage_matrix = hierarchical_model_selection(eigval, dist=lambda l1, l2: (l1 - l2) / l1, return_linkage_matrix=True)
    BIC = dict([(model, bic_fast(eigval, eigvec, mu, n, p, model)) for model in models])
    best_model = sorted(BIC, key=BIC.get, reverse=False)[0]

    labels = np.zeros((p,))
    for j in range(p):
        if j in np.cumsum(best_model):
            labels[j:] += 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    n_clusters = 1 + int(np.max(labels))
    ax1.bar(np.arange(0, p), eigval, color=plt.cm.get_cmap('coolwarm')(1 - labels / np.max(labels)))
    dendrogram(linkage_matrix, color_threshold=0, ax=ax2)
    plt.savefig(path + "hierarchical_clustering.png", dpi='figure', format='png', transparent=True)
    plt.savefig(path + "hierarchical_clustering.pdf", dpi='figure', format='pdf', transparent=True)
    plt.savefig(path + "hierarchical_clustering.svg", dpi='figure', format='svg', transparent=True)
    plt.close()
