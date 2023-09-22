import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    np.random.seed(42)
    path = os.path.dirname(__file__) + "/figures/"

    plt.figure(figsize=(8, 8))
    plt.rc('grid', linestyle="--", color='black', alpha=.5)
    n_list = (np.logspace(1, 6.0, num=1000))
    plt.loglog(2 * (1 - np.exp(2 * np.log(n_list) / n_list) + np.sqrt(np.exp(4 * np.log(n_list) / n_list) - np.exp(2 * np.log(n_list) / n_list))), n_list, linewidth=5)
    plt.ylabel('Number of samples')
    plt.xlabel('Relative eigengap')
    plt.grid(True, which='both')
    plt.savefig(path + f"relative_eigengap.png", dpi='figure', format='png', transparent=True)
    plt.savefig(path + f"relative_eigengap.pdf", dpi='figure', format='pdf', transparent=True)
    plt.savefig(path + f"relative_eigengap.svg", dpi='figure', format='svg', transparent=True)
    plt.close()