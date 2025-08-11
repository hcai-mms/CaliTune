import argparse
import pandas as pd
from tqdm import tqdm


def compute_user_pop_baselines(train_split_path, popularity_path, user_embedding_path):
    """
    Generate user-specific popularity baselines and save them to a TSV file.

    Args:
        train_split_path (str): Path to the training split file (TSV format).
        popularity_path (str): Path to the popularity file (TSV format).
    """
    train_split_df = pd.read_csv(train_split_path, sep='\t')
    popularity_df = pd.read_csv(popularity_path, sep='\t')
    user_embeddings_df = pd.read_csv(user_embedding_path, sep='\t')
    output_file = popularity_path.replace("popularity", "userbaseline")

    # Initialize an empty dataframe to store the results
    result_df = pd.DataFrame(columns=['user_id'] + list(popularity_df.columns[1:]))

    # Iterate over each user in the train_split_df
    # Intentionally do not sort the groupby because we need to ensure the same order as in the training interaction
    # file (and thus the user embeddings in Recbole)
    for user_id in tqdm(range(user_embeddings_df.shape[0]), desc="Computing user pop baselines"):

        items = train_split_df[train_split_df['user_id'] == user_id]['item_id'].astype(str)
        total_items = len(items)
        pop_entries = popularity_df.iloc[items.values]
        # sum up all columns
        distribution = pop_entries.iloc[:, 1:].sum(axis=0)

        # Convert counts to distribution by dividing by the total number of items
        if total_items != 0:
            distribution /= total_items
        else:
            user_id = '[PAD]'

        distribution['user_id'] = str(user_id)
        result_df = result_df._append(distribution, ignore_index=True)

    # Save the result dataframe to a TSV file
    result_df.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    """
    Parse command-line arguments and generate user popularity baselines.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--train_split', type=str, default='splits/ml-100k/train_split.tsv')
    parser.add_argument('-p', '--popularity', type=str, default='data/ml-100k/ml-100k.popularity')
    args = parser.parse_args()

    compute_user_pop_baselines(args.train_split, args.popularity)
    print("User popularity baselines have been saved.")
