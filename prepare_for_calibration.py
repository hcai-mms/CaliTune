import argparse
import os

from preparation_pipeline.embeddings import get_and_save_embeddings
from preparation_pipeline.popularity_items import compute_item_popularity
from preparation_pipeline.popularity_users import compute_user_pop_baselines
from preparation_pipeline.train_test_splits import record_splits


def main(args):
    # Dynamically derive paths from the dataset name
    dataset_path = args.dataset_path
    checkpoint_path = args.checkpoint_path
    dataset = args.dataset
    inter_path = os.path.join("data", dataset, f"{args.dataset}.inter")
    user_embedding_path = os.path.join("data", dataset, f"{args.dataset}.useremb")
    item_embedding_path = os.path.join("data", dataset, f"{args.dataset}.itememb")
    train_split_path = os.path.join("splits", dataset, "train_split.tsv")
    popularity_path = os.path.join("data", dataset, f"{args.dataset}.popularity")

    # Step 1: Generate embeddings
    print("Step 1: Generating embeddings...")
    get_and_save_embeddings(dataset_path, dataset, checkpoint_path)

    # Step 2: Generate data splits
    print("Step 2: Generating data splits...")
    record_splits(args.dataset, checkpoint_path)

    # Step 3: Generate item popularity
    print("Step 3: Generating item popularity...")
    compute_item_popularity(inter_path, item_embedding_path, args.pop_proportions)

    # Step 4: Generate user popularity baselines
    print("Step 4: Generating user popularity baselines...")
    compute_user_pop_baselines(train_split_path, popularity_path, user_embedding_path)

    print("All steps completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares the environment for calibration by generating necessary files from a given baseline Recbole Model.")

    # Argument for dataset
    parser.add_argument('-d', '--dataset', type=str, default='ml-100k',
                        help="Name of the dataset folder (e.g., 'ml-100k'). Relative to the 'data' directory.")

    parser.add_argument('-c', '--checkpoint_path', type=str, default='saved/ml-100k-256.pth',
                        help="Path to the model checkpoint file to use as a baseline")

    parser.add_argument('-n', '--dataset_path', type=str, default='data',
                        help="Path to the dataset directory. Default is 'data'.")

    # Argument for popularity proportions (used in get_popularity)
    parser.add_argument('-p', '--pop_proportions', type=float, nargs='+', default=[0.2, 0.6, 0.2],
                        help="List of popularity mass for each popularity bin group (must sum to 1)")

    args = parser.parse_args()

    # Check if proportions sum to 1
    if not 0.99 <= sum(args.pop_proportions) <= 1.01:
        print("Error: The populariy proportions must sum to 1.")
        exit(1)

    main(args)
