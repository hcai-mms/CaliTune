import numpy as np
import torch
from tqdm import trange, tqdm
import math


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL Divergence between two probability distributions using PyTorch.

    Args:
    p (Tensor): Probability distribution P.
    q (Tensor): Probability distribution Q.

    Returns:
    Tensor: The KL Divergence for each row.
    """
    return torch.sum(p * torch.log2(p / q), dim=1)


def compute_jsd(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jensen-Shannon Divergence (JSD) between two probability distributions using PyTorch.

    Args:
    prediction (Tensor): Normalized prediction tensor (probability distribution).
    target (Tensor): Normalized target tensor (probability distribution).

    Returns:
    Tensor: The JSD value for each sample.
    """
    epsilon = 1e-10  # Small value to avoid log(0)

    # Ensure the input tensors are valid probability distributions
    prediction = prediction + epsilon
    target = target + epsilon

    # Compute the mean distribution
    m = 0.5 * (prediction + target)

    # Compute the KL divergence between each distribution and the mean distribution
    kl_pm = kl_divergence(prediction, m)
    kl_qm = kl_divergence(target, m)

    # Compute the JSD as the average of the KL divergences
    jsd = 0.5 * (kl_pm + kl_qm)

    return jsd  # Return JSD for each sample


def get_jsd(prediction, target, valid_batch_size):
    """
    Compute the JSD value for each sample over all items, processed in batches using PyTorch.

    Args:
    prediction (Tensor): Normalized prediction tensor (probability distribution) of shape (N, num_classes).
    target (Tensor): Normalized target tensor (probability distribution) of shape (N, num_classes).
    valid_batch_size (int): Batch size for processing the data.

    Returns:
    Tensor: The JSD values for each sample.
    """
    all_jsd = []

    num_batches = (prediction.size(0) + valid_batch_size - 1) // valid_batch_size  # Calculate number of batches

    for i in trange(num_batches, desc="JSD", leave=False):
        start_idx = i * valid_batch_size
        end_idx = min((i + 1) * valid_batch_size, prediction.size(0))

        batch_prediction = prediction[start_idx:end_idx]
        batch_target = target[start_idx:end_idx]

        batch_jsd = compute_jsd(batch_prediction, batch_target)
        all_jsd.append(batch_jsd)

    # Concatenate all the JSD values from batches into a single tensor
    return torch.cat(all_jsd, dim=0)


def get_scores_for_all(max_user_id, device, model, positive_train):
    users = torch.arange(max_user_id + 1, device=device)
    scores = model.full_predict(users)
    scores = scores.detach()
    scores[positive_train == 1] = -torch.inf
    return users, scores


def get_ideal_dcg(y_true: torch.Tensor, k: int) -> torch.Tensor:
    exp_decay = 1.0 / torch.log2(torch.arange(2, k + 2))
    # ideal ndcg is the cumulative sum of the exponential decay up to that point
    ideal_dcg_per_count = torch.zeros(k + 1)
    ideal_dcg_per_count[1:] = torch.cumsum(exp_decay, dim=0)

    y_true_count = torch.count_nonzero(y_true, dim=1)
    y_true_count = torch.clamp(y_true_count, 0, k)

    # map each entry in y_true_count to the respective index in ideal_dcg_per_count
    ideal_dcg = ideal_dcg_per_count[y_true_count]

    return ideal_dcg


def binary_NDCG(logits: torch.Tensor, y_true: torch.Tensor, k: int, ideal_dcg: torch.Tensor, device: torch.device):
    exp_decay = (1.0 / torch.log2(torch.arange(2, k + 2))).to(device)

    _, idx_best_scores = torch.topk(logits, k, dim=1, largest=True, sorted=True)
    # compute dcg per user using the y_true matrix at indices idx_best_scores
    dcgs = torch.sum(y_true.gather(1, idx_best_scores) * exp_decay, dim=1)

    ndcgs = torch.where(ideal_dcg != 0, dcgs / ideal_dcg, torch.zeros_like(dcgs))
    return ndcgs


def get_ndcg(scores: torch.Tensor, actual: torch.Tensor, ndcg_batch_size, k, device):
    ideal_dcgs = get_ideal_dcg(actual, k)

    ndcgs = torch.zeros(scores.shape[0], device=device)
    for i in trange(
        0, scores.shape[0], ndcg_batch_size,
        desc=f"NDCG",
        leave=False,
        dynamic_ncols=True,
        smoothing=0,
    ):
        start = i
        end = min(scores.shape[0], i + ndcg_batch_size)

        # batching NDCG to conserve memory usage
        ndcgs[start:end] = binary_NDCG(
            logits=scores[start:end, :].to(device),
            y_true=actual[start:end, :].to(device),
            k=k,
            ideal_dcg=ideal_dcgs[start:end].to(device),
            device=device
        )
    return ndcgs


def evaluate(
        config,
        to_test: bool, to_report: bool,
        max_user_id: int,
        device: torch.device,
        model,
        y_true_train: torch.Tensor,
        y_true_valid: torch.Tensor,
        y_true_test: torch.Tensor,
        jsd_target: torch.Tensor,
        item_popularity: torch.Tensor,
        logger
):
    # drop padding user
    user_popularity_targets = jsd_target[1:]

    valid_batch_size = config['valid_batch_size']
    top_k = config['top_k_ndcg']
    remove_valid_items = config['remove_valid_items']

    model.eval()
    users, scores = get_scores_for_all(max_user_id, device, model, y_true_train)

    # calculate NDCG, disregard the padding user
    valid_ndcg = get_ndcg(scores, y_true_valid, valid_batch_size, top_k, device)[1:]
    valid_ndcg = valid_ndcg.mean()
    if to_report:
        logger.info(f"Validation NDCG: {valid_ndcg}")

    # remove recommendations contained in validation/test set (for testing recbole removes items from validation)
    if remove_valid_items:
        scores[y_true_valid == 1] = -torch.inf

    test_ndcg = None
    if to_test:
        # [1:] --> disregard the padding user
        test_ndcg_all = get_ndcg(scores, y_true_test, valid_batch_size, top_k, device)[1:]
        test_ndcg = test_ndcg_all.mean()
        if to_report:
            logger.info(f"Test NDCG: {test_ndcg}")

    # Calculate JSD (@k)
    top_k_item_ids = []


    POP_DIST_BATCH_SIZE = 1000
    for i in range(0, len(users), POP_DIST_BATCH_SIZE):
        start_idx = i
        end_idx = min(i + POP_DIST_BATCH_SIZE, len(users))

        batch_scores = scores[start_idx:end_idx].to(device)
        _, top_k_indices = torch.topk(batch_scores, top_k, dim=1, largest=True, sorted=True)

        top_k_item_ids.append(top_k_indices)

    top_k_item_ids = torch.cat(top_k_item_ids, dim=0)

    item_popularity = item_popularity.to(device)
    top_k_popularity = item_popularity[top_k_item_ids]  # (6040, 10, 3)

    top_k_popularity_sum = torch.sum(top_k_popularity, dim=1)  # (6040, 3)
    top_k_popularity_sum = top_k_popularity_sum[1:]  # ignore Padding User

    top_k_popularity_dist = top_k_popularity_sum / (top_k_popularity_sum.sum(dim=1, keepdim=True) + 1e-10)

    # Also drop padding user here
    # user_popularity_targets = convert_to_tensor(jsd_target[1:], device)
    user_popularity_targets = user_popularity_targets.to(device)
    jsd_all = get_jsd(top_k_popularity_dist, user_popularity_targets, valid_batch_size)
    jsd = jsd_all.mean()
    if to_report:
        logger.info(f"JSD: {jsd}")

    return valid_ndcg, test_ndcg, jsd
