import matplotlib.pyplot as plt
import numpy as np
import os

from inference import maximum_log_likelihood, model_dimension
from model_selection import generate_models


def run_bic_trajectories(mean, S_pop, models, n_space, nb_indep_space, compare_PPCA=False):
    bic_results = []
    bic_mean = []
    bic_model_choice_percentage = []
    if compare_PPCA:
        bic_model_choice_percentage_PPCA = []
        PPCA_models = generate_models(S_pop.shape[0], family="PPCA")
        indexes_PPCA = [models.index(model) for model in PPCA_models]
    for i, (n, nb_indep) in enumerate(zip(n_space, nb_indep_space)):
        print(i / len(n_space))
        bic_n = np.zeros((nb_indep, len(models)))
        for j in range(nb_indep):
            X = np.random.multivariate_normal(mean, S_pop, size=n // 1)
            for k, model in enumerate(models):
                dim = model_dimension(model)
                bic = 1 / n * (dim * np.log(n) - 2 * maximum_log_likelihood(X, model))
                bic_n[j, k] = bic
        bic_n_mean = np.mean(bic_n, axis=0)
        bic_model_choice = np.argmin(bic_n, axis=1)
        unique, counts = np.unique(bic_model_choice, return_counts=True)
        bic_n_model_choice_percentage = np.zeros(len(models))
        for l, model in enumerate(unique):
            bic_n_model_choice_percentage[model] = counts[l] / nb_indep

        if compare_PPCA:
            bic_model_choice_f = np.argmin(bic_n[:, indexes_PPCA], axis=1)
            unique_f, counts_f = np.unique(bic_model_choice_f, return_counts=True)
            bic_n_model_choice_percentage_f = np.zeros(len(PPCA_models))
            for l, model in enumerate(unique_f):
                bic_n_model_choice_percentage_f[model] = counts_f[l] / nb_indep
            bic_model_choice_percentage_PPCA.append(bic_n_model_choice_percentage_f)
        bic_results.append(bic_n)
        bic_mean.append(bic_n_mean)
        bic_model_choice_percentage.append(bic_n_model_choice_percentage)

    bic_mean = np.array(bic_mean)
    bic_model_choice_percentage = np.array(bic_model_choice_percentage)
    bic_model_choice_percentage_PPCA = np.array(bic_model_choice_percentage_PPCA)

    return bic_results, bic_mean, bic_model_choice_percentage, bic_model_choice_percentage_PPCA


if __name__ == '__main__':
    np.random.seed(42)
    path = os.path.dirname(__file__) + "/figures/"

    # RUN BIC EXPERIMENT
    p = 5
    mean = np.random.rand(p)
    Sigma_pop = [10, 9, 7, 4, .5]
    S_pop = np.array(np.diag(Sigma_pop))
    eigval, eigvec = np.linalg.eigh(S_pop)
    nb_n = 50
    min_log_n = np.log10(20)
    max_log_n = np.log10(50000)
    n_space = np.logspace(min_log_n, max_log_n, num=nb_n).astype('int')
    nb_indep_space = np.logspace(4, 2, nb_n).astype('int')
    PPCA_models = generate_models(p, family="PPCA")
    SPCA_models_dim = dict([(model, model_dimension(model)) for model in generate_models(p, family="SPCA")])
    SPCA_models = sorted(SPCA_models_dim, key=SPCA_models_dim.get, reverse=False)
    models = SPCA_models
    bic_results, bic_mean, bic_model_choice_percentage, bic_model_choice_percentage_f = run_bic_trajectories(mean, S_pop, models, n_space, nb_indep_space, compare_PPCA=True)

    # PLOT
    vmin, vmax = min(SPCA_models_dim.values()), max(SPCA_models_dim.values())
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 9]})
    ax1.bar(np.arange(1, p + 1), np.flip(eigval, -1))
    ax1.set_title(f'Eigenprofile', fontsize=15)

    for i in range(nb_n - 1):
        max_model = models[np.argmax(bic_model_choice_percentage[i])]
        color = sm.to_rgba(SPCA_models_dim[max_model])
        color = np.clip(color, 0., 1.)
        ax2.axvspan(n_space[i], n_space[i + 1], facecolor=color, alpha=0.5)
    for i, model in enumerate(models):
        if model in PPCA_models:
            ax2.plot(n_space, bic_mean[:, i], label=str(model), color=sm.to_rgba(SPCA_models_dim[model]), linestyle='dashed')
        else:
            ax2.plot(n_space, bic_mean[:, i], label=str(model), color=sm.to_rgba(SPCA_models_dim[model]))
    for i, model in enumerate(models):
        ax2.scatter(n_space, bic_mean[:, i], s=150 * (bic_model_choice_percentage[:, i]) ** 2, marker='o', color=sm.to_rgba(SPCA_models_dim[model]), edgecolors='black')
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of samples', fontsize=15)
    ax2.set_ylabel('BIC mean', fontsize=15)
    ax2.set_title('Mean BIC and percentage of model selection as a function of the number of samples', fontsize=15)
    cbar = plt.colorbar(sm)
    cbar.set_label('Model Dimension', fontsize=15)
    cbar.remove()
    ax2.legend(prop={'size': 15})
    plt.tight_layout()
    exp_name = "_".join(str((np.array(Sigma_pop)*100).astype('int'))[1:-1].split())
    plt.savefig(path + f"synthetic_exp.png", dpi='figure', format='png', transparent=True)
    plt.savefig(path + f"synthetic_exp.pdf", dpi='figure', format='pdf', transparent=True)
    plt.savefig(path + f"synthetic_exp.svg", dpi='figure', format='svg', transparent=True)
    plt.close()

    plt.figure(figsize=(8, 8))
    complexity_SPCA_hard = np.array([model_dimension(model) for model in SPCA_models])[np.argmax(bic_model_choice_percentage, axis=1)]
    complexity_PPCA_hard = np.array([model_dimension(model) for model in PPCA_models])[np.argmax(bic_model_choice_percentage_f, axis=1)]
    complexity_SPCA_weighted = bic_model_choice_percentage @ np.array([SPCA_models_dim[model] for model in SPCA_models])
    complexity_PPCA_weighted = bic_model_choice_percentage_f @ np.array([SPCA_models_dim[model] for model in PPCA_models])
    plt.plot(n_space, complexity_PPCA_hard, color='tab:red', linewidth=5, label="PPCA - most selected")
    plt.plot(n_space, complexity_PPCA_weighted, color='tab:red', linewidth=5, ls='--', label="PCA - averaged")
    plt.plot(n_space, complexity_SPCA_hard, color='tab:blue', linewidth=5, label="SPCA - most selected")
    plt.plot(n_space, complexity_SPCA_weighted, color='tab:blue', linewidth=5, ls='--', label="SPCA - averaged")
    plt.xscale('log')
    plt.xlabel('Number of samples')
    plt.ylabel('Model dimension')
    plt.title('Evolution of model dimension')
    plt.legend()
    plt.savefig(path + f"synthetic_exp_dim.png", dpi='figure', format='png', transparent=True)
    plt.savefig(path + f"synthetic_exp_dim.pdf", dpi='figure', format='pdf', transparent=True)
    plt.savefig(path + f"synthetic_exp_dim.svg", dpi='figure', format='svg', transparent=True)
    plt.close()
