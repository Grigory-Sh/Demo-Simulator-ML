import numpy as np
from typing import Tuple
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
    ) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
    y_pred = classifier.predict_proba(X)[:,1]
    roc_auc = []
    while n_bootstraps > 0:
        ind = np.random.choice(len(y), len(y))
        if len(set(y[ind])) < 2:
            continue
        roc_auc.append(roc_auc_score(y[ind], y_pred[ind]))
        n_bootstraps -= 1
    lcb, ucb = np.quantile(roc_auc, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return (lcb, ucb)
