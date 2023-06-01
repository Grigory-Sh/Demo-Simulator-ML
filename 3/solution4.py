from typing import List
import numpy as np

def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """Average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values ​​\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    
    def count_score(lst: List[float], method: str):
        count = 0
        for i, val in enumerate(lst):
            if method == 'standard':
                count += val / np.log2(i + 2)
            if method == 'industry':
                count += (np.power(2, val) - 1) / np.log2(i + 2)
        return count
    
    score = 0
    for relevance in list_relevances:
        dcg = count_score(relevance[:k], method)
        idcg = count_score(np.sort(relevance)[::-1][:k], method)
        if idcg != 0:
            score += dcg / idcg
    score /= len(list_relevances)
    return score
