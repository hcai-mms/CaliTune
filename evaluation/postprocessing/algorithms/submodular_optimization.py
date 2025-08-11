import numpy as np
import numba as nb
from ..utils.distribution import jensen_shannon
from typing import List, Callable


# complexity: let k = num recs, n = num items, l = num users
# -> roughly k*n*l evaluations of objective function for k << n
# if k=n then (n^2 + n)/2 * l evaluations,(since we first do n loops, then n-1 loops ans so on -> gauß formula)
# -> poly growth
# -> better do k << n,
# reordering entire list probably not necessary -> use just m << n top items

def greedy(candidates: np.ndarray, rels: np.ndarray, n: int, Q_fn: Callable, P_dist: np.ndarray,
           compute_utility: Callable, lam: float = .5):
    calibrated_recs: List[np.uint32] = []
    for i in range(n):
        max_util = -np.inf
        best_item_index = None
        for j in candidates:
            if j in calibrated_recs:
                continue

            Lu = np.array(calibrated_recs + [j], dtype=np.uint32)
            Q_dist = Q_fn(Lu)

            util = compute_utility(Lu, Q_dist, P_dist, rels, lam)

            if util > max_util:
                max_util = util
                best_item_index = j

        calibrated_recs.append(best_item_index)
    return calibrated_recs


@nb.njit
def js_calibration(Lu: list, Q_dist: np.ndarray, P_dist: np.ndarray, preds: np.ndarray, lam: float = .5):
    rel = np.sum(preds[Lu])
    return (1 - lam) * rel - lam * jensen_shannon(P_dist, Q_dist)


@nb.njit
def js_calibration_average(Lu: list, Q_dist: np.ndarray, P_dist: np.ndarray, preds: np.ndarray, lam: float = .5):
    rel = np.sum(preds[Lu]) / len(Lu)
    return (1 - lam) * rel - lam * jensen_shannon(P_dist, Q_dist)
