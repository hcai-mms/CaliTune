import numpy as np
import pandas as pd
import os
from scipy import sparse
import torch
import csv
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .postprocessing.postprocess import execute_postprocessing


def load_and_transform_tsv(filepath: str) -> np.ndarray:
    df = pd.read_csv(filepath, sep="\t")

    group_positions = df.values
    values = group_positions

    return values


def load_embeddings(
        embeddings_path: str, device: torch.device
) -> tuple[list, str, str, torch.Tensor]:
    """Load embeddings directly to specified device."""
    id_list = []
    embeddings = []
    with open(embeddings_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader)
        id_col, emb_col = headers[0], headers[1]

        for row in reader:
            uid = row[0].replace("[PAD]", "0")
            id_list.append(int(uid))
            emb = [float(x) for x in row[1].split()]
            embeddings.append(emb)

    return (
        id_list,
        id_col,
        emb_col,
        torch.tensor(embeddings, dtype=torch.float, device=device),
    )


def _get_mask_indices(
        sparse_matrix: csr_matrix, start_idx: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Convert sparse matrix indices to GPU tensor for batch processing."""
    n_items = sparse_matrix.shape[1]
    indices = []

    for i in range(batch_size):
        global_idx = start_idx + i
        row_start = sparse_matrix.indptr[global_idx]
        row_end = sparse_matrix.indptr[global_idx + 1]
        indices.extend(
            (sparse_matrix.indices[row_start:row_end] + n_items * i).tolist()
        )

    return torch.tensor(indices, dtype=torch.long, device=device) if indices else None


def get_preds(user_emb, item_emb, train_hist, sparse_hist, device):
    size = user_emb.size(0)

    scores = torch.mm(user_emb, item_emb.t())
    scores[0, :] = -torch.inf
    scores[:, 0] = -torch.inf

    sparse_hist = sparse_hist.tocsr()
    # Pre-calculate mask indices
    mask = _get_mask_indices(sparse_hist, 0, size, device)
    train_mask = _get_mask_indices(train_hist, 0, size, device)

    scores.view(-1)[mask] = -torch.inf
    scores.view(-1)[train_mask] = -torch.inf

    return scores


def get_preds_batched(user_emb, item_emb, sparse_hist, device, batch_size):
    user_emb = user_emb
    item_emb = item_emb

    allscores = []
    num_users = user_emb.size(0)

    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)
        user_batch = user_emb[start:end]

        scores = torch.mm(user_batch, item_emb.t())
        size = user_batch.size(0)

        sparse_hist = sparse_hist.tocsr()
        # Pre-calculate mask indices
        train_mask = _get_mask_indices(sparse_hist, start, size, device)

        scores.view(-1)[train_mask] = -torch.inf
        allscores.append(scores)

    return torch.stack(allscores)


def main_postprocessing(base_path, experiment_name, item_catgories_fname, model_name):
    n = 10
    m = 100
    lams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prebinned = load_and_transform_tsv(
        f"{base_path}/{experiment_name}/{item_catgories_fname}"
    )
    prebinned = prebinned[:, 1:].astype(np.float32)

    hist_train = sparse.load_npz(f"{base_path}/train_hist.npz")
    hist_valid = sparse.load_npz(f"{base_path}/valid_hist.npz")

    attribute = "popularity"

    os.makedirs(
        os.path.join(f"{base_path}/{experiment_name}", "postprocessed"),
        exist_ok=True,
    )

    for eval_target in ["test"]:
        baseline_predictions = np.load(
            f"{base_path}/base_scores_{eval_target}.npy"
        )


        for lam in tqdm(lams, desc='Executing Postprocessing for different lambdas'):
            preds_processed = execute_postprocessing(
                preds=baseline_predictions,
                history_train=hist_train,
                attribute=attribute,
                n=n,
                objective_fun="jensen-shannon",
                lam=lam,
                m=m,
                use_n_hist=False,
                history_test=None,
                normalize=True,
                user_lams=None,
                categories=prebinned,
            )
            np.save(
                f"{base_path}/{experiment_name}/postprocessed/result_{lam}_{eval_target}.npy",
                preds_processed,
            )