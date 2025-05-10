from collections import defaultdict
import numpy as np

class KnowledgeBase:
    def __init__(self):
        self.db = defaultdict(list)

    def record(self, distribution_name, loss):
        self.db[distribution_name].append(loss)

    def best_distribution(self):
        avg_losses = {k: np.mean(v) for k, v in self.db.items()}
        return min(avg_losses, key=avg_losses.get)