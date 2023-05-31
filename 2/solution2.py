import numpy as np

def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    """Loss function for LifeTime Value"""

    sum_errors = 0
    num = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_t > y_p:
            sum_errors += (np.log(y_t + 1) - np.log(y_p + 1)) ** 2
        else:
            sum_errors += (np.log(y_t + 1) - np.log(y_p + 1)) ** 2 * 10
        num += 1
    error = np.sqrt(sum_errors / num)
    return error
