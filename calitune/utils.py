import os
from typing import Any

import numpy as np
import pandas as pd
import torch

import yaml


def read_config(config_path: str) -> dict[str, Any]:
    with open(config_path, 'r') as file:
        user_config = yaml.safe_load(file)

    return user_config


def get_cuda_device(config, logger):
    try:
        if config['device'] is None:
            raise ValueError
        else:
            device = torch.device(config['device'])
            logger.info(f'Using CUDA device specified in config: {device}')
            return device
    except Exception:
        print('Device specified in config could not be found.')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA is available, using GPU")
            return torch.device("cuda")  # Use CUDA device if available
        else:
            logger.info("CUDA is available, using CPU")
            return torch.device("cpu")


def set_seed(seed_value):
    torch.manual_seed(seed_value)  # PyTorch random seed
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    np.random.seed(seed_value)  # Numpy module.
    torch.backends.cudnn.deterministic = True  # To ensure deterministic algorithm is used
    torch.backends.cudnn.benchmark = False


def load_embeddings(embeddings_path: str) -> tuple[list, str, str, torch.Tensor]:
    """Load embeddings from the given tsv file. List of IDs and column names are also returned to later replicate the
    format

    Args:
        embeddings_path (str): Path to tsv file with embeddings

    Returns:
        tuple[list,str,str,torch.Tensor]: A tuple containing:
            - list: List of IDs
            - str: Name of ID column
            - str: name of embedding column
            - torch.Tensor: embeddings as tensor

    """
    df = pd.read_csv(embeddings_path, sep='\t')
    # Dynamically read column names
    id_column_name = df.columns[0]  # Assuming the first column is ID
    embedding_column_name = df.columns[1]  # Assuming the second column contains the lists of numbers

    # Replace '[PAD]' in the ID column with 0
    df[id_column_name] = df[id_column_name].replace('[PAD]', 0)
    df[id_column_name] = pd.to_numeric(df[id_column_name])
    # save the original order of the embeddings
    id_list = df[id_column_name].tolist()

    # Convert the strings of numbers into lists of numbers
    df[embedding_column_name] = df[embedding_column_name].apply(lambda x: [float(num) for num in x.split()])

    # Extract the lists into a numpy array and then convert to a tensor
    embedding_tensor = torch.tensor(df[embedding_column_name].tolist(), dtype=torch.float)
    return id_list, id_column_name, embedding_column_name, embedding_tensor


def fix_missing_items(df, N):
    count = 0
    for idx in range(1, N + 1):
        if idx not in df.index:
            df.loc[idx] = [0]
            count += 1
    df.sort_index(inplace=True)  # Ensure the indices are sorted
    return count


def load_splits(split_folder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load the list of positive item IDs for each split

    Args:
        split_folder (str): Path to the folder containing the tsv files for the splits

    Returns:
        tuple[ndarray, ndarray, ndarray,int,int]: A tuple containing:
            - 3 ndarrays: positive item IDs for train, validation and test splits
            - int: max_item_id: highest item ID
            - int: max_user_id: highest user ID
    """
    train_df = pd.read_csv(os.path.join(split_folder, 'train_split.tsv'), sep='\t', usecols=['user_id', 'item_id'])
    valid_df = pd.read_csv(os.path.join(split_folder, 'valid_split.tsv'), sep='\t', usecols=['user_id', 'item_id'])
    test_df = pd.read_csv(os.path.join(split_folder, 'test_split.tsv'), sep='\t', usecols=['user_id', 'item_id'])
    # Concatenate all DataFrames
    combined_df = pd.concat([train_df, valid_df, test_df])

    # Find unique users and items
    unique_users = combined_df['user_id'].nunique()
    unique_items = combined_df['item_id'].nunique()

    # Group by 'user_id' and collect 'item_id' into lists
    grouped_train = train_df.groupby('user_id')['item_id'].apply(list)
    grouped_valid = valid_df.groupby('user_id')['item_id'].apply(list)
    grouped_test = test_df.groupby('user_id')['item_id'].apply(list)

    # In case a user does not have any items for validation or test, add empty list
    N = len(grouped_train)
    if len(grouped_valid) < N:
        count = fix_missing_items(grouped_valid, N)
        print(f"{count} users do not have validation items!")

    if len(grouped_test) < N:
        count = fix_missing_items(grouped_test, N)
        print(f"{count} users do not have test items!")

    # Convert the Series of lists into a NumPy array of objects (each object is a list)
    return (np.array(grouped_train), np.array(grouped_valid),
            np.array(grouped_test), int(unique_items), int(unique_users))


def load_popularity(popularity_file: str):
    values = pd.read_csv(popularity_file, sep='\t').values[:, 1:].astype(np.int8)

    return torch.from_numpy(values)


def generate_candidates(model, top_k, max_user_id, device, batch_size=1000):
    users = torch.arange(max_user_id + 1, device=device)
    scores = model.full_predict(users)

    indices = []

    for i in range(0, len(users), batch_size):
        start_idx = i
        end_idx = min(i + batch_size, len(users))

        batch_scores = scores[start_idx:end_idx].to(device)
        _, top_k_indices = torch.topk(batch_scores, k=top_k, dim=1, largest=True, sorted=False)
        indices.append(top_k_indices.cpu().detach().numpy())

    return np.concatenate(indices, axis=0)


def select_candidates(config: dict[str, Any], model, max_user_id: int, max_item_id: int, device: torch.device) -> np.ndarray:
    top_k = config['k_candidates']
    if config['candidate_selection'] == 'random':
        return np.random.randint(1, max_item_id + 1, size=(max_user_id + 1, top_k))
    elif config['candidate_selection'] == 'relevance':
        return generate_candidates(model, top_k, max_user_id, device)
    else:
        raise ValueError(f'Unknown candidate selection method: {config["candidate_selection"]}')


def load_candidate_score_targets(y_true: torch.Tensor, candidates: np.ndarray) -> torch.Tensor:
    row_indices = np.arange(y_true.shape[0])[:, np.newaxis]
    return y_true[row_indices, candidates]


def load_popularity_targets(user_baseline_file):
    baseline_df = pd.read_csv(user_baseline_file, sep='\t')
    baseline = baseline_df.values[:, 1:]
    # Create padding row
    # num_columns = baseline.shape[1]
    # padding_row = np.zeros(num_columns)

    # Insert the padding row
    # padded_baseline = np.insert(baseline, 0, padding_row, axis=0)
    return torch.from_numpy(baseline.astype(np.float32))


def get_targets_for_splits(positive, num_users, num_items):
    # Initialize the new 2D array with zeros
    actual = np.zeros((num_users + 1, num_items + 1), dtype=np.int8)

    # Populate the new array based on indices from positive_valid
    for user_id, item_ids in enumerate(positive):
        # Set specified indices to 1
        actual[user_id + 1, [x for x in item_ids]] = 1

    return torch.from_numpy(actual)


def move_dict_to_device(data_dict: dict, device: torch.device) -> dict:
    """
    Moves all tensors in a dictionary to the specified device.

    Args:
        data_dict (dict): A dictionary where values are tensors.
        device (torch.device): The device to move the tensors to.

    Returns:
        dict: A new dictionary with all tensors moved to the device.
    """
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in data_dict.items()}


def save_embedding(data_path, dataset, embeddings: np.ndarray, id_list: list, id_column_name: str,
                   embedding_column_name: str, ending=None, extension=None, save_path=None) -> None:
    """Save embeddings in the same format as it was read from file

    Args:
        save_path: saved embedding path
        embeddings (numpy.ndarray): Embeddings as numpy array
        id_list (list): list of IDs
        extension (str): extension to save as. ".useremb" for useremb embeddings, ".itememb" for item embeddings
        id_column_name (str): name of ID column
        embedding_column_name (str): name of embedding column

    Returns:
        None

    """
    # Convert each embedding array into a space-separated string
    # TODO: this is rather ineffficient. Is there anything against properly saving the embeddings as a numpy array?
    embeddings_as_strings = [' '.join(map(str, embedding)) for embedding in embeddings]

    # Replace ID 0 with '[PAD]'
    id_list_modified = [id if id != 0 else '[PAD]' for id in id_list]

    # Create a DataFrame with the modified ID list and embeddings strings
    df = pd.DataFrame({
        id_column_name: id_list_modified,
        embedding_column_name: embeddings_as_strings
    })

    # Save to TSV
    if save_path is None:
        save_path = os.path.join(data_path, dataset, f'{dataset}-{ending}.{extension}')
    df.to_csv(str(save_path), sep='\t', index=False)


def restrict_scores(predictions):
    min_val = predictions.min()
    max_val = predictions.max()
    return (predictions - min_val) / (max_val - min_val + 1e-10)  # Add epsilon to avoid division by zero
