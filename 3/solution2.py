from typing import List
import numpy as np

def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    score = 0
    for i, val in enumerate(relevance[:k]):
        if method == 'standard':
            score += val / np.log2(i + 2)
        if method == 'industry':
            score += (np.power(2, val) - 1) / np.log2(i + 2)
    return score
