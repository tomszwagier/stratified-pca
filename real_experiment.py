import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from inference import bic
from model_selection import generate_models, hierarchical_model_selection, type_length_model_selection

if __name__ == '__main__':
    np.random.seed(42)
    dataset = "wdbc"  # [wine, glass, ionosphere, wdbc]

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), f"data/{dataset}.data"), sep=",")
    data = data[~(data == '?').any(axis=1)]
    if dataset == "wine":
        data_X, data_y = data.to_numpy()[:, 1:], data.to_numpy()[:, 0]
        data_X = data_X[data_y == 3]
    elif dataset == "glass":
        data_X, data_y = data.to_numpy()[:, 1:-1].astype('float'), data.to_numpy()[:, -1]
        data_X = data_X[data_y == 3]
    elif dataset == "ionosphere":
        data_X, data_y = data.to_numpy()[:, 2:-1].astype('float'), data.to_numpy()[:, -1]
        data_X = data_X[data_y == 'g']
    elif dataset == "wdbc":
        data_X, data_y = data.to_numpy()[:, 2:].astype('float'), data.to_numpy()[:, 1]
        data_X = data_X[data_y == 'B']
    else:
        raise NotImplementedError(f"The dataset {dataset} does not exist.")

    data_X = data_X.reshape((data_X.shape[0], -1))
    data_X = (data_X - np.mean(data_X, axis=0))
    if dataset in ["wdbc", "wine"]:
        data_X /= np.std(data_X, axis=0)

    n, p = data_X.shape
    S = (1 / n) * data_X.T @ data_X
    eigval, _ = np.linalg.eigh(S)
    eigval = np.flip(eigval, -1)
    eigval = np.clip(eigval, 0, np.inf)
    relative_eigengaps = (eigval[:-1] - eigval[1:]) / eigval[:-1]
    print(relative_eigengaps)

    models_SPCA = hierarchical_model_selection(eigval, dist=lambda l1, l2: (l1 - l2) / l1)
    for model in models_SPCA:
        print(f"{model}  -- BIC = {bic(data_X, model):.2f}")
    models_PPCA = generate_models(p, "PPCA")
    for model in models_PPCA:
        print(f"{model}  -- BIC = {bic(data_X, model):.2f}")
    print("\n==========\n")
    for cardinal in [1, 2, 3, 4, p]:
        model, BIC = type_length_model_selection(data_X, cardinal, criterion="BIC")
        print(f"{model}  -- BIC = {BIC:.2f}")
        model = (1,) * (cardinal - 1) + (p - (cardinal - 1),)
        print(f"{model}  -- BIC = {bic(data_X, model):.2f}")









