import torch
import numpy as np

def generate_dataset(n_samples=1000, n_features=20, seed=42):
    np.random.seed(seed)
    X = np.random.rand(n_samples, n_features)
    y = (X.sum(axis=1) > n_features / 2).astype(int)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)