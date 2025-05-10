import numpy as np
import random
from sklearn.metrics import roc_auc_score
import torch

def ensemble_predict(models, X):
    with torch.no_grad():
        outputs = [model(X) for model in models]
        probs = torch.stack([torch.softmax(out, dim=1) for out in outputs])
        return probs.mean(dim=0)

def shapley_values(models, X, y, n_samples=50):
    n = len(models)
    shapley = np.zeros(n)
    for _ in range(n_samples):
        perm = list(range(n))
        random.shuffle(perm)
        prev_auc = 0
        ensemble = []
        for i, idx in enumerate(perm):
            ensemble.append(models[idx])
            y_prob = ensemble_predict(ensemble, X)
            auc = roc_auc_score(y.numpy(), y_prob[:, 1].numpy())
            shapley[idx] += auc - prev_auc
            prev_auc = auc
    return shapley / n_samples

def prune_models(models, shapley_values, threshold=0.05):
    max_val = np.max(shapley_values)
    selected = [m for i, m in enumerate(models) if shapley_values[i] >= max_val * threshold]
    return selected