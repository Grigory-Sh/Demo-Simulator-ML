import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    '''Loss function for business problem'''
    underforecast = np.mean(((y_true - y_pred) / y_pred) ** 2)
    reforecast = np.mean(((y_true - y_pred) / y_true) ** 2)
    error = min(underforecast, reforecast)
    return error
