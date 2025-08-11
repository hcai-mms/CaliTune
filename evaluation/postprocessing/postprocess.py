import numpy as np
from .utils.distribution import get_dist_fun
from .algorithms.submodular_optimization import greedy, js_calibration, js_calibration_average
from typing import Callable
from tqdm import trange


def postprocess(
    preds: np.ndarray,
    history_train: np.ndarray,
    n: int,
    dist_fun: Callable,
    objective_fun: Callable,
    lam: float = 0.5,
    m: int = None,
    algorithm: str = "so_greedy",
    use_n_hist: bool = False,
    history_test: np.ndarray = None,
    categories: np.ndarray = None,
    user_lams: np.ndarray = None,
):
    if history_train.shape != preds.shape:
        raise ValueError("history and prediction shapes do not match")

    # -1 in case use_n_hist is used bc the list can then be <n
    results = np.ones((preds.shape[0], n), dtype=np.int32) * -1

    for i in trange(1, preds.shape[0]):
        uh = history_train[i].toarray()[0]
        
        (inter_idx,) = uh.nonzero()
        
        P_dist = dist_fun(inter_idx, categories)

        Q_fun = lambda x: dist_fun(x, categories)

        # if use_n_hist, use length of history as n for better evaluation later on (but at most m)
        if not use_n_hist:
            nu = n
        else:
            if history_test is None:
                raise ValueError(
                    "if 'use_n_hist is true, a history for test set needs to be provided'"
                )
            nu = np.min((len(history_test[i].nonzero()[0]), m))

        if user_lams is not None:
            lam = user_lams[i]

        if lam is None:
            raise ValueError("Lambda cannot be None")
        
        # if m then only consider top m items for postprocessing (performance)
        candidates = np.argsort(-preds[i])[:m]
        if algorithm == "so_greedy":
            opt_pred_indices = greedy(
                candidates,
                preds[i].astype(np.float32),
                nu,
                Q_fun,
                P_dist,
                objective_fun,
                lam,
            )
            results[i] = opt_pred_indices
        else:
            raise ValueError("No valid algorithm for postprocessing provided")

    return results


def execute_postprocessing(
    preds: np.ndarray,
    history_train: np.ndarray,
    attribute: str,
    n: int,
    objective_fun: str,
    lam: float = 0.5,
    m: int = None,
    use_n_hist: bool = False,
    history_test=None,
    normalize: bool = False,
    user_lams: np.ndarray = None,
    categories: np.ndarray = None,
):
    dist_fun = get_dist_fun(attribute, categories)

    if objective_fun == "jensen-shannon":
        objective_fun = js_calibration
    elif objective_fun == "jensen-shannon-average":
        objective_fun = js_calibration_average
    else:
        raise ValueError("no valid objective function provided")
    
    if normalize:
        min_val = np.min(preds[preds != -np.inf])
        preds -= min_val

        max_val = np.max(preds)
        preds /= max_val
    
    preds = postprocess(
        preds,
        history_train,
        n,
        dist_fun,
        objective_fun,
        lam,
        m,
        use_n_hist=use_n_hist,
        history_test=history_test,
        categories=categories,
        user_lams=user_lams,
    )

    return preds
