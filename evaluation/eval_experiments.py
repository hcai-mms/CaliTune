import numpy as np
import torch

from .postprocessing.utils.distribution import category_dist_torch
import pandas as pd

import json
import os
from scipy import sparse
from tqdm import trange


def load_user_baseslines(user_baseline_file):
    baseline_df = pd.read_csv(user_baseline_file, sep="\t")
    # Drop user id column
    baseline = baseline_df.values[:, 1:]
    # Create padding row
    # num_columns = baseline.shape[1]
    # padding_row = np.zeros(num_columns)

    # Insert the padding row
    # padded_baseline = np.insert(baseline, 0, padding_row, axis=0)
    return baseline.astype(np.float32)


def load_item_popularities(filepath: str) -> np.ndarray:
    df = pd.read_csv(filepath, sep="\t")
    values = df.iloc[:, 1:].values

    return values.astype(np.int8)


def split_user_groups(jsd_target: np.ndarray, splits=None) -> list[np.ndarray]:
    """
    Partition the sorted indices of jsd_target into bins defined by splits.

    Args:
        jsd_target (np.ndarray): A 2D numpy array where partitioning is based on column 0.
        splits (list of float): A list of fractions that sum to 1 (e.g., [0.2, 0.6, 0.2]).

    Returns:
        list of np.ndarray: A list where each element is the set of indices corresponding to a bin.
    """
    if splits is None:
        # Default fallback
        splits = [0.2, 0.6, 0.2]

    if not np.isclose(sum(splits), 1.0):
        raise ValueError("The split fractions must sum to 1.0")

    # Drop the padding user, as it will also not be reported in the JSD and NDCG metrics
    sorted_indices = (-jsd_target[1:, 0]).argsort()
    n = len(sorted_indices)

    # Compute cumulative indices using rounding
    boundaries = [0]
    for split in splits:
        boundaries.append(boundaries[-1] + split * n)
    # Convert boundaries to integer indices (last boundary becomes n)
    boundaries = [int(round(b)) for b in boundaries]
    boundaries[-1] = n  # Ensure the last index is exactly n

    groups = []
    for i in range(len(splits)):
        groups.append(sorted_indices[boundaries[i]:boundaries[i + 1]])

    return groups


def obtain_metrics(ndcgs: torch.Tensor, jsds: torch.Tensor, user_groups):
    ret = {
        'ndcg': ndcgs.mean().item(),
        'jsd': jsds.mean().item(),
    }

    for i, group in enumerate(user_groups):
        ret[f'ndcg_usergroup_{i + 1}'] = ndcgs[group].mean().item()
        ret[f'jsd_usergroup_{i + 1}'] = jsds[group].mean().item()

    return ret


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


def get_jsd(
        prediction: torch.Tensor,
        target: torch.Tensor,
        device,
        batch_size: int = 500,
) -> torch.Tensor:
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

    num_batches = (prediction.size(0) + batch_size - 1) // batch_size  # Calculate number of batches

    for i in trange(num_batches, desc="JSD", leave=False):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, prediction.size(0))

        batch_prediction = prediction[start_idx:end_idx].to(device)
        batch_target = target[start_idx:end_idx].to(device)

        batch_jsd = compute_jsd(batch_prediction, batch_target)
        all_jsd.append(batch_jsd)

    jsds = torch.cat(all_jsd, dim=0)
    return jsds


def get_ideal_dcg(y_true: torch.Tensor, k: int) -> torch.Tensor:
    exp_decay = 1.0 / torch.log2(torch.arange(2, k + 2))
    # ideal ndcg is the cumulative sum of the exponential decay up to that point
    ideal_dcg_per_count = torch.zeros(k + 1).to(y_true.device)
    ideal_dcg_per_count[1:] = torch.cumsum(exp_decay, dim=0)

    y_true_count = torch.count_nonzero(y_true, dim=1)
    y_true_count = torch.clamp(y_true_count, 0, k)

    # map each entry in y_true_count to the respective index in ideal_dcg_per_count
    ideal_dcg = ideal_dcg_per_count[y_true_count]

    return ideal_dcg


def ndcg_at_k(recommended_items: torch.Tensor, y_true: torch.Tensor, ideal_dcg: torch.Tensor, k: int) -> torch.Tensor:
    exp_decay = (1.0 / torch.log2(torch.arange(2, k + 2, device=recommended_items.device)))

    recommended_items = recommended_items[:, :k]

    dcgs = torch.sum(y_true.gather(1, recommended_items) * exp_decay, dim=1)
    ndcgs = torch.where(ideal_dcg != 0, dcgs / ideal_dcg, torch.zeros_like(dcgs))
    return ndcgs


def get_ndcg(
        predictions: torch.Tensor,
        y_true: torch.Tensor,
        k: int,
        device: torch.device,
        batch_size=500,
) -> torch.Tensor:
    user_count = predictions.shape[0]
    ndcg_scores = []

    ideal_dcg = get_ideal_dcg(y_true, k)

    for start_idx in trange(0, user_count, batch_size, desc="NDCG", leave=False):
        end_idx = min(start_idx + batch_size, user_count)
        predicted_items = predictions[start_idx:end_idx].to(device)
        relevant_items = y_true[start_idx:end_idx].to(device)
        ideal_scores = ideal_dcg[start_idx:end_idx].to(device)
        ndcgs = ndcg_at_k(predicted_items, relevant_items, ideal_scores, k)
        ndcg_scores.append(ndcgs)

    ndcgs = torch.cat(ndcg_scores, dim=0)
    return ndcgs


def evaluate(
        pred_indices,
        jsd_target,
        item_popularities,
        ndcg_target_hist,
        k=10,
        dist_fun=category_dist_torch,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # immediately drop padding user
    prediction_tensor = torch.from_numpy(pred_indices[1:]).to(torch.int64)
    y_true_tensor = torch.from_numpy(ndcg_target_hist[1:])
    jsd_target_tensor = torch.from_numpy(jsd_target[1:])
    item_categories_tensor = torch.from_numpy(item_popularities)

    ndcgs = get_ndcg(prediction_tensor, y_true_tensor, k, device)
    torch.cuda.empty_cache()

    pred_dist = dist_fun(prediction_tensor, item_categories_tensor)
    jsds = get_jsd(pred_dist, jsd_target_tensor, device)
    torch.cuda.empty_cache()

    return ndcgs, jsds


################params
def evaluate_experiment(
        base_path: str,
        experiment_name: str,
        per_user_metrics: bool = False
):
    finetuned_base_path = f"{base_path}/{experiment_name}/"
    for folder in [
        f
        for f in os.listdir(finetuned_base_path)
        if os.path.isdir(os.path.join(finetuned_base_path, f)) and f != "postprocessed"
    ]:
        evaluate_experiment_folder(base_path, experiment_name, folder, "test", per_user_metrics)


def evaluate_experiment_folder(base_path, experiment_name, experiment_run_folder, mode, per_user_metrics: bool = False):
    experiment_path = f"{experiment_name}/{experiment_run_folder}"
    print(experiment_path)

    with open(f"{base_path}/{experiment_path}/{mode}.config.json") as f:
        config = json.load(f)

    prebinned = load_item_popularities(config["item_popularity"])
    jsd_target = load_user_baseslines(config["popularity_target"])

    user_groups = split_user_groups(jsd_target)

    base_recs = np.load(config["base_recommendations"]).astype(np.int32)

    hist_target = (
        sparse.load_npz(config["hist_test"]).toarray().astype(np.int8)
        if mode == "test"
        else sparse.load_npz(config["hist_valid"]).toarray().astype(np.int8)
    )

    dist_fun = category_dist_torch

    result = {}
    per_user_results = pd.DataFrame(columns=[
        'model',
        'user_id',
        'userbaseline_group_1', 'userbaseline_group_2', 'userbaseline_group_3',
        'ndcg', 'jsd'
    ])

    ################execution
    ndcgs, jsds = evaluate(
        base_recs, jsd_target, prebinned, hist_target, dist_fun=dist_fun,
    )
    result["base_recommendations"] = obtain_metrics(ndcgs, jsds, user_groups)

    if per_user_metrics:
        # Append per-user NDCG and JSD to the DataFrame. Epoch here is -1 to mark the base model
        per_user_results = pd.concat([per_user_results, pd.DataFrame({
            'model': 'base_model',
            'user_id': np.arange(1, len(jsds) + 1),
            'userbaseline_group_1': jsd_target[1:, 0],
            'userbaseline_group_2': jsd_target[1:, 1],
            'userbaseline_group_3': jsd_target[1:, 2],
            'ndcg': ndcgs.cpu().numpy(),
            'jsd': jsds.cpu().numpy()
        })], ignore_index=True)

    print("no post processing:")
    print(f"ndcg: {result['base_recommendations']['ndcg']}")
    print(f"js divergence: {result['base_recommendations']['jsd']}")

    # for rec_group in ["postprocessed", "finetuned"]:
    for rec_group in ["postprocessed", "finetuned"]:
        result[rec_group] = {}
        for key, path in config[rec_group].items():
            try:
                recs = np.load(path).astype(np.int32)
            except OSError:
                print(f"File not found: {path}! Skipping...")
                continue

            ndcgs, jsds = evaluate(
                recs, jsd_target, prebinned, hist_target, dist_fun=dist_fun,
            )
            result[rec_group][key] = obtain_metrics(ndcgs, jsds, user_groups)
            if per_user_metrics:
                # Append per-user NDCG and JSD to the DataFrame
                per_user_results = pd.concat([per_user_results, pd.DataFrame({
                    'model': f'{rec_group}_{key}',
                    'user_id': np.arange(1, len(jsds) + 1),
                    'userbaseline_group_1': jsd_target[1:, 0],
                    'userbaseline_group_2': jsd_target[1:, 1],
                    'userbaseline_group_3': jsd_target[1:, 2],
                    'ndcg': ndcgs.cpu().numpy(),
                    'jsd': jsds.cpu().numpy()
                })], ignore_index=True)

            print(f"key={key}")
            print(f"ndcg: {result[rec_group][key]['ndcg']}")
            print(f"js divergence: {result[rec_group][key]['ndcg']}")

    with open(
            f"{base_path}/{experiment_path}/result_{experiment_run_folder}_{mode}.json", "w"
    ) as f:
        json.dump(result, f, indent=2)

    if per_user_metrics:
        per_user_results.to_csv(
            f"{base_path}/{experiment_path}/per_user_results_{experiment_run_folder}_{mode}.tsv.zip",
            sep='\t',
        )

    print("done")


if __name__ == "__main__":
    base_path = "data/ML-1M_tuned_ep_all/experiments"
    experiment_name = "GENRE"

    evaluate_experiment(base_path, experiment_name)
