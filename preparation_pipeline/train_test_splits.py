"""
Load a trained model and save data split IDs to tsv files
"""

from recbole.quick_start.quick_start import load_data_and_model
import argparse
import pandas as pd
import os


def extract_user_item_pairs(data,user_col, item_col):
    user_item_pairs = pd.DataFrame({
        'user_id': data.interaction[user_col].numpy(),
        'item_id': data.interaction[item_col].numpy()
    })
    return user_item_pairs.groupby('user_id')['item_id'].apply(list)


def save_split(split, path, filename):
    data = []
    for user_id, item_ids in split.items():
        for item_id in item_ids:
            data.append({'user_id': user_id, 'item_id': item_id})
    df = pd.DataFrame(data)

    # Save the DataFrame to a TSV file
    df.to_csv(os.path.join(str(path), filename) + ".tsv", sep='\t', index=False)


def record_splits(dataset_name, checkpoint_path):
    # Load trained model and data from recbole
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=checkpoint_path
    )

    user_col = dataset.uid_field
    item_col = dataset.iid_field

    train = extract_user_item_pairs(train_data.dataset.inter_feat, user_col, item_col)
    valid = extract_user_item_pairs(valid_data.dataset.inter_feat, user_col, item_col)
    test = extract_user_item_pairs(test_data.dataset.inter_feat, user_col, item_col)

    path = os.path.join('splits', dataset_name)

    os.makedirs(path, exist_ok=True)

    save_split(train, path, "train_split")
    save_split(valid, path, "valid_split")
    save_split(test, path, "test_split")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='ml-100k')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='saved/ml-100k_base.pth')
    args = parser.parse_args()

    record_splits(args.dataset, args.checkpoint_path)

    print(f"Splits saved to {str(os.path.join('splits', args.dataset))}")
