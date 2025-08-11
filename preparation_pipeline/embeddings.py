"""
Load a trained model and save the embeddings to tsv files
"""

import os

from recbole.model.general_recommender import BPR, DMF
from recbole.quick_start.quick_start import load_data_and_model
import argparse
import numpy as np
import torch


def save_to_tsv(filename, ids, embeddings, header):
    """
    Save IDs and their corresponding embeddings to a TSV file.

    Args:
        filename (str): Path to the TSV file to save.
        ids (list): List of IDs (user or item) to save.
        embeddings (ndarray): Embedding vectors corresponding to the IDs.
        header (str): Header row for the TSV file.
    """
    # Create the folder if it doesn't exist yet
    folder_path = os.path.dirname(filename)
    os.makedirs(folder_path, exist_ok=True)

    with open(filename, 'w') as f:
        f.write(header + "\n")
        for uid, emb in zip(ids, embeddings):
            emb_str = ' '.join(map(str, emb))
            f.write(f"{uid}\t{emb_str}\n")


def get_and_save_embeddings(dataset_path: str, dataset_name: str, checkpoint_path: str):
    """
    Load a trained model and extract user and item embeddings,
    saving them to TSV files.

    Args:
        dataset_path (str): Path to the dataset directory.
        checkpoint_path (str): Path to the model checkpoint file.
    """
    # Load trained model and data from recbole
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=checkpoint_path
    )

    if isinstance(model, DMF):
        # get_user_embedding() simply returns the user embeddings of the first layer which are then passed into the MLP layers.
        # These can be directly saved.
        user_embeddings = model.get_user_embedding(torch.arange(0, dataset.user_num)).detach().cpu().numpy()

        # The method get_item_embedding() returns the item embeddings of the last layer after the MLP layers.
        # Instead, we need copy-paste of the first part of the implementation of get_item_embedding() to get the item embeddings
        # we need. See recbole.model.general_recommender.dmf.DMF.get_item_embedding() for the source
        interaction_matrix = model.interaction_matrix.tocoo()
        row = interaction_matrix.row
        col = interaction_matrix.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(interaction_matrix.data)
        item_matrix = (
            torch.torch.sparse_coo_tensor(i, data, torch.Size(interaction_matrix.shape))
            .to(model.device)
            .transpose(0, 1)
        )
        item_embeddings = torch.sparse.mm(item_matrix, model.item_linear.weight.t()).detach().cpu().numpy()
    elif isinstance(model, BPR):
        user_embeddings = model.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = model.item_embedding.weight.detach().cpu().numpy()

    torch.save(model.state_dict(), os.path.join(dataset_path, dataset_name, f'base_weights.pth'))

    # Get external user and item IDs
    external_user_ids = dataset.id2token(dataset.uid_field, range(0, dataset.user_num))
    external_item_ids = dataset.id2token(dataset.iid_field, range(0, dataset.item_num))

    # Save user and item embeddings to TSV files
    save_to_tsv(f'{os.path.join(dataset_path, dataset_name, dataset_name)}.useremb', external_user_ids[:],
                user_embeddings,
                "uid:token\tuser_emb:float_seq")
    save_to_tsv(f'{os.path.join(dataset_path, dataset_name, dataset_name)}.itememb', external_item_ids[
                                                                                     :], item_embeddings,
                "iid:token\titem_emb:float_seq")

    print(f'User and item embeddings have been saved to {os.path.join(dataset_path, dataset_name)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, default='data')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='saved/ml-100k_base.pth')
    args = parser.parse_args()

    get_and_save_embeddings(args.dataset_path, args.checkpoint_path)
