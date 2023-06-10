from typing import Tuple
import numpy as np

def pr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
    ) -> Tuple[float, float]:
    """Returns threshold and recall (from Precision-Recall Curve)"""
    tp = np.cumsum(y_true == 1)
    threshold_proba = y_prob[0]
    max_recall = tp[0] / tp[-1]
    for i in range(1, y_true.shape[0]):
        precision = tp[i] / (i + 1)
        if precision >= min_precision:
            recall = tp[i] / tp[-1]
            if recall > max_recall:
                max_recall = recall
                threshold_proba = y_prob[i]
    return threshold_proba, max_recall

def sr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float,
    ) -> Tuple[float, float]:
    """Returns threshold and recall (from Specificity-Recall Curve)"""
    tp = np.cumsum(y_true == 1)
    threshold_proba = y_prob[0]
    max_recall = tp[0] / tp[-1]
    num = y_true.shape[0] - tp[-1]
    left = -1
    right = y_true.shape[0]
    while left + 1 < right:
        middle = (left + right) // 2
        specificity = (num - middle - 1 + tp[middle]) / num
        if specificity < min_specificity:
            right = middle
        elif specificity > min_specificity:
            left = middle
        else:
            left = middle
            break
    max_recall = tp[left] / tp[-1]
    threshold_proba = y_prob[left]
    return threshold_proba, max_recall

def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Precision-Recall curve and it's (LCB, UCB)"""
    num = y_true.shape[0]
    tp = np.cumsum(y_true == 1)
    recall = np.array([0.0])
    precision = np.array([1.0])
    precision_lcb = np.array([1.0])
    precision_ucb = np.array([1.0])
    recall = np.append(recall, tp / tp[-1])
    serial_number = [i + 1 for i in range(y_true.shape[0])]
    precision = np.append(precision, tp / serial_number)
    precisions = []
    while n_bootstrap > 0:
        ind = np.sort(np.random.choice(num, num))
        if len(set(y_true[ind])) < 2:
            continue
        tp_sample = np.cumsum(y_true[ind] == 1)
        precision_sample = tp_sample / serial_number
        precision_interpolated = np.maximum.accumulate(precision_sample[::-1])[::-1]
        precisions.append(precision_interpolated)
        n_bootstrap -= 1
    lcb, ucb = np.quantile(precisions, [(1 - conf) / 2, 1 - (1 - conf) / 2], axis=0)
    precision_lcb = np.append(precision_lcb, lcb)
    precision_ucb = np.append(precision_ucb, ucb)
    return recall, precision, precision_lcb, precision_ucb

def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Specificity-Recall curve and it's (LCB, UCB)"""
    num = y_true.shape[0]
    tp = np.cumsum(y_true == 1)
    zeros_nums = np.cumsum(y_true == 0)
    n = zeros_nums[-1]
    tn = n - zeros_nums
    recall = np.array([0.0])
    specificity = np.array([1.0])
    specificity_lcb = np.array([1.0])
    specificity_ucb = np.array([1.0])
    recall = np.append(recall, tp / tp[-1])
    specificity = np.append(specificity, tn / n)
    specificitys = []
    while n_bootstrap > 0:
        ind = np.sort(np.random.choice(num, num))
        if len(set(y_true[ind])) < 2:
            continue
        zeros_nums_sample = np.cumsum(y_true[ind] == 0)
        n_sample = zeros_nums_sample[-1]
        tn_sample = n_sample - zeros_nums_sample
        specificity_sample = tn_sample / n_sample
        specificity_interpolated = np.maximum.accumulate(specificity_sample[::-1])[::-1]
        specificitys.append(specificity_interpolated)
        n_bootstrap -= 1
    lcb, ucb = np.quantile(specificitys, [(1 - conf) / 2, 1 - (1 - conf) / 2], axis=0)
    specificity_lcb = np.append(specificity_lcb, lcb)
    specificity_ucb = np.append(specificity_ucb, ucb)
    return recall, specificity, specificity_lcb, specificity_ucb
