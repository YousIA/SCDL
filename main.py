import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from dataset import generate_dataset
from model import SimpleNet
from train import train_model
from shapley import shapley_values, prune_models, ensemble_predict
from knowledge_base import KnowledgeBase

def scdl_pipeline():
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    models = [SimpleNet(X.shape[1]) for _ in range(5)]
    models = [train_model(m, X_train, y_train) for m in models]

    shap_values = shapley_values(models, X_test, y_test)
    print("Shapley values:", shap_values)

    pruned_models = prune_models(models, shap_values, threshold=0.2)
    print(f"Models after pruning: {len(pruned_models)}")

    kb = KnowledgeBase()
    loss_fn = nn.CrossEntropyLoss()

    for dist in ["normal", "noisy", "shifted"]:
        X_varied, y_varied = generate_dataset(seed={"normal": 42, "noisy": 43, "shifted": 44}[dist])
        y_pred = ensemble_predict(pruned_models, X_varied)
        loss = loss_fn(y_pred, y_varied).item()
        kb.record(dist, loss)

    best_dist = kb.best_distribution()
    print(f"Best data distribution based on loss: {best_dist}")

    final_preds = ensemble_predict(pruned_models, X_test)
    auc = roc_auc_score(y_test.numpy(), final_preds[:, 1].numpy())
    print(f"Final AUC: {auc:.4f}")

if __name__ == "__main__":
    scdl_pipeline()