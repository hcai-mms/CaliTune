import argparse
import numpy as np
import pandas as pd
import sys


def load_and_rename_interaction_data(inter_path):
    """
    Load interaction data from a file and rename specific columns for consistency.

    Args:
        inter_path (str): Path to the interaction data file (TSV format).

    Returns:
        pd.DataFrame: The interaction data with renamed columns.
    """
    # Load the data
    interaction_data = pd.read_csv(inter_path, sep='\t')

    # Rename specific columns and adjust others
    rename_mapping = {}
    for col in interaction_data.columns:
        if col == 'user_id:token':
            rename_mapping[col] = 'user'
        elif col == 'item_id:token':
            rename_mapping[col] = 'item'
        else:
            rename_mapping[col] = col.split(':')[0]

    interaction_data.rename(columns=rename_mapping, inplace=True)

    return interaction_data


def custom_segment_items(interaction_data, proportions):
    """
    Segment items into groups based on their popularity.

    Args:
        interaction_data (pd.DataFrame): Interaction data containing user-item interactions.
        proportions (list[float]): Proportions for dividing items into groups (must sum to 1).

    Returns:
        pd.DataFrame: A DataFrame with items and their assigned groups.
    """
    # Calculate item popularity by counting the number of unique users for each item
    item_popularity = interaction_data.groupby('item')['user'].nunique().sort_values(ascending=False)

    # Get the cumulative counts to split into groups
    total_unique_users = item_popularity.sum()
    cumulative_counts = np.cumsum(proportions) * total_unique_users

    segmented_data = pd.DataFrame({'item': item_popularity.index, 'group': np.zeros(len(item_popularity))})

    # Assign group numbers based on cumulative counts
    current_index = 0
    for i, count in enumerate(cumulative_counts):
        next_index = np.searchsorted(item_popularity.cumsum(), count)
        segmented_data.loc[current_index:next_index, 'group'] = i + 1
        current_index = next_index + 1

    return segmented_data


def load_item_embedding(item_embedding_path):
    """
    Load item embeddings from a file and extract item IDs.

    Args:
        item_embedding_path (str): Path to the item embedding file (TSV format).

    Returns:
        list[int]: A list of item IDs extracted from the embedding file.
    """
    # Load the item embedding file
    item_embedding_df = pd.read_csv(item_embedding_path, sep='\t')
    id_column_name = item_embedding_df.columns[0]

    # Create the item_id_list
    item_id_list = item_embedding_df[id_column_name].tolist()
    if item_id_list[0] == '[PAD]':
        item_id_list = item_id_list[1:]

    item_id_list = list(map(int, item_id_list))

    return item_id_list


def compute_item_popularity(inter_path, item_embedding_path, proportions):
    """
    Generate item popularity segmentation and save it to a file.

    Args:
        inter_path (str): Path to the interaction data file (TSV format).
        item_embedding_path (str): Path to the item embedding file (TSV format).
        proportions (list[float]): Proportions for dividing items into groups (must sum to 1).
    """
    num_groups = len(proportions)
    interaction_data = load_and_rename_interaction_data(inter_path)
    segmented_data = custom_segment_items(interaction_data, np.array(proportions))

    popularity_matrix = pd.DataFrame(index=segmented_data.index)
    popularity_matrix['item'] = segmented_data['item']

    # Dynamically create columns based on the number of groups
    for i in range(num_groups):
        group_name = f'Group_{i + 1}'
        popularity_matrix[group_name] = 0
        popularity_matrix.loc[segmented_data['group'] == i + 1, group_name] = 1

    # Sort the popularity_matrix by item_id_list
    item_id_list = load_item_embedding(item_embedding_path)
    popularity_matrix['item'] = popularity_matrix['item'].astype(int)
    popularity_matrix = popularity_matrix.set_index('item').reindex(item_id_list).reset_index()

    # Make sure padding is still the first row
    pad_row = pd.DataFrame({'item': '[PAD]'}, index=[0])
    for i in range(num_groups):
        pad_row[f'Group_{i + 1}'] = 0

    final_popularity_matrix = pd.concat([pad_row, popularity_matrix], ignore_index=True)

    popularity_file_path = inter_path[:-5] + "popularity"
    final_popularity_matrix.to_csv(popularity_file_path, sep='\t', index=False)
    print(f'Item popularity segmentation has been saved to {popularity_file_path}')


if __name__ == '__main__':
    """
    Parse command-line arguments and run the item popularity segmentation process.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inter_path', type=str, default="data/ml-100k/ml-100k.inter",
                        help="Path to the interaction data file")
    parser.add_argument('-e', '--item_embedding_path', type=str, default="data/ml-100k/ml-100k.itememb",
                        help="Path to the item embedding file")
    parser.add_argument('-p', '--proportions', type=float, nargs='+', default=[0.2, 0.6, 0.2],
                        help="List of proportions for each group")
    args = parser.parse_args()

    # Check if the proportions sum to 1
    if not np.isclose(sum(args.proportions), 1.0):
        print("Error: The proportions must sum to 1.")
        sys.exit(1)

    compute_item_popularity(args.inter_path, args.item_embedding_path, args.proportions)
