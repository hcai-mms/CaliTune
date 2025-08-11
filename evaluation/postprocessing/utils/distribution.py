import numpy as np
import numba as nb
from typing import Literal, Tuple, List
import torch

QUOTE_NONE = 3


@nb.njit
def _KL(x: np.ndarray, y: np.ndarray, base: Literal[2, np.e] = 2) -> float:
    eps = 1e-10

    a = x + eps
    a /= a.sum()

    b = y + eps
    b /= b.sum()

    if base == 2:
        return float(np.sum(a * np.log2(a / b)))
    return float(np.sum(a * np.log(a / b)))


# [0,1] see https://arxiv.org/pdf/2008.05932.pdf
@nb.njit
def jensen_shannon(P, Q, base=2):
    M = (P + Q) / 2
    return (_KL(P, M, base=base) + _KL(Q, M, base=base)) / 2

@nb.njit
def category_dist(rec_indices: np.ndarray, item_categories: np.ndarray):
    #indices are 1d array
    rec_categories = item_categories[rec_indices]
    category_sums = rec_categories.sum(axis=0)

    category_dist = category_sums / category_sums.sum()

    return category_dist

def get_dist_fun(attribute: str, categories: np.ndarray = None):
    if attribute == "popularity":
        dist_fun = category_dist
    elif attribute == "country":
        dist_fun = category_dist
    else:
        raise ValueError("no valid attribute provided")
    return dist_fun

def category_dist_torch(rec_indices: torch.Tensor, item_categories: torch.Tensor):
    #indices are 1d array
    rec_categories = item_categories[rec_indices]
    category_sums = rec_categories.sum(dim=1)

    category_dist = category_sums / category_sums.sum(dim=1, keepdim=True)

    return category_dist