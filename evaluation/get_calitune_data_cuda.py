import os

import numpy as np
import torch
from recbole.data import data_preparation, create_dataset
from recbole.model.general_recommender import BPR, DMF
from recbole.utils import init_seed, get_model
from scipy import sparse

from calitune.models import CaliDMF, CaliBPR


def load_recbole_data_and_model(model_file: str):
    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)

    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return model, train_data, valid_data, test_data


def obtain_scores(
        model,
        train_inter: np.ndarray,
        valid_inter: np.ndarray,
        test_inter: np.ndarray,
):
    scores = model.full_predict(
        torch.arange(0, test_inter.shape[0]), batch_size=2048
    )

    scores[train_inter == 1] = -torch.inf

    valid_scores = scores.clone().detach()
    valid_scores[test_inter == 1] = -torch.inf

    test_scores = scores.clone().detach()
    test_scores[valid_inter == 1] = -torch.inf

    return valid_scores, test_scores

def obtain_recommendations(
    model,
    train_inter: np.ndarray,
    valid_inter: np.ndarray,
    test_inter: np.ndarray,
):
    valid_scores, test_scores = obtain_scores(model, train_inter, valid_inter, test_inter)
    topk_valid = torch.topk(valid_scores, 10, dim=1).indices
    topk_test = torch.topk(test_scores, 10, dim=1).indices

    return topk_valid.cpu().numpy(), topk_test.cpu().numpy()


def save_transform_calitune_data(
    base_path,
    model_name,
    experiment_name,
):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recbole_model, train_data, valid_data, test_data = load_recbole_data_and_model(
        model_file=os.path.join(base_path, model_name)
    )

    # Save sparse interaction matrices
    def save_sparse(matrix, path):
        sparse_matrix = matrix.dataset.inter_matrix().tocsr()
        sparse.save_npz(path + ".npz", sparse_matrix)

    save_sparse(train_data, os.path.join(base_path, "train_hist"))
    save_sparse(valid_data, os.path.join(base_path, "valid_hist"))
    save_sparse(test_data, os.path.join(base_path, "test_hist"))


    train_inter = train_data.dataset.inter_matrix().toarray()
    valid_inter = valid_data.dataset.inter_matrix().toarray()
    test_inter = test_data.dataset.inter_matrix().toarray()

    # Load model and datasets
    if isinstance(recbole_model, BPR):
        model = CaliBPR(os.path.join(base_path, 'base_weights.pth'), torch.from_numpy(train_inter))
    elif isinstance(recbole_model, DMF):
        model = CaliDMF(os.path.join(base_path, 'base_weights.pth'), torch.from_numpy(train_inter))
    else:
        raise ValueError(f'Unsupported Model class: {type(recbole_model)}')

    base_scores_valid, base_scores_test = obtain_scores(model, train_inter, valid_inter, test_inter)
    base_recs_valid, base_recs_test = obtain_recommendations(model, train_inter, valid_inter, test_inter)

    # Save base scores and recommendations
    np.save(os.path.join(base_path, "base_scores_valid.npy"), base_scores_valid)
    np.save(os.path.join(base_path, "base_scores_test.npy"), base_scores_test)
    np.save(os.path.join(base_path, "base_recs_valid.npy"), base_recs_valid)
    np.save(os.path.join(base_path, "base_recs_test.npy"), base_recs_test)

    # Process fine-tuned models
    finetuned_base_path = os.path.join(base_path, experiment_name)
    for parent in [
        f
        for f in os.listdir(finetuned_base_path)
        if os.path.isdir(os.path.join(finetuned_base_path, f))
    ]:
        finetuned_path = os.path.join(finetuned_base_path, parent)
        for folder in [
            f
            for f in os.listdir(finetuned_path)
            if os.path.isdir(os.path.join(finetuned_path, f))
        ]:
            print('Generating Recommendations for:', folder)
            finetuned_weights = torch.load(os.path.join(finetuned_path, folder, 'finetuned_params.pth'), map_location=device)
            model.load_state_dict(finetuned_weights)

            finetuned_recs_valid, finetuned_recs_test = obtain_recommendations(model, train_inter, valid_inter, test_inter)
            # Save results
            np.save(
                os.path.join(finetuned_path, folder, "preds_finetuned_valid.npy"),
                finetuned_recs_valid,
            )
            np.save(
                os.path.join(finetuned_path, folder, "preds_finetuned_test.npy"),
                finetuned_recs_test,
            )


if __name__ == "__main__":
    save_transform_calitune_data(
        base_path="data/ML-1M_tuned_ep_all/experiments/",
        model_name="BPR-Jan-29-2025_11-59-07_ml-1m_tuned_256.pth",
        experiment_name="ADDITIONAL_POP",
    )

