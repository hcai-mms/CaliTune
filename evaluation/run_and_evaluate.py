import argparse
import os

from .main_postprocessing import main_postprocessing
from .get_calitune_data_cuda import save_transform_calitune_data
from .create_configs import create_configs
from .eval_experiments import evaluate_experiment


def run_and_evaluate(
    base_path,
    model_name,
    experiment_name,
    item_catgories_fname,
    user_basline_fname,
    compute_baseline=True,
    per_user_metrics=False,
):
    # generates recommendations and saves interaction matrices etc.
    save_transform_calitune_data(
        base_path, model_name, experiment_name
    )

    # executes postprocessing
    if compute_baseline:
        print("start postprocessing")
        main_postprocessing(base_path, experiment_name, item_catgories_fname, model_name)
    else:
        os.makedirs(
            os.path.join(f"{base_path}/{experiment_name}", "postprocessed"),
            exist_ok=True,
        )

    # creates json configs for evaluation
    print("creating eval configs")
    create_configs(base_path, experiment_name, item_catgories_fname, user_basline_fname)

    # evalautes experiments and saves results into experiment folders where data came from
    print("start evaluating")
    evaluate_experiment(base_path, experiment_name, per_user_metrics)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--base_path', type=str, default="data")
    parser.add_argument('-d', '--dataset', type=str, default="ml-100k")
    parser.add_argument('-m', '--base_model', type=str, default="ml-100k-256.pth")
    parser.add_argument('-e', '--experiment_name', type=str, default="example1")
    parser.add_argument('-b', '--compute_baseline', type=bool, action=argparse.BooleanOptionalAction,
                        help="Set explicitly to True if you're interpreting many experiments and only need the postprocessing once in order to save time")
    parser.add_argument('-u', '--per_user_metrics', type=bool, action=argparse.BooleanOptionalAction,
                        help="Set to True for additional per-user metrics")
    parser.add_argument('--batch_size', type=int, default=2048)
    args = parser.parse_args()

    dataset_path = os.path.join(args.base_path, args.dataset)
    item_catgories_fname = f'{args.dataset}.popularity'
    user_basline_fname = f'{args.dataset}.userbaseline'
    user_emb = f'{args.dataset}.useremb'
    item_emb = f'{args.dataset}.itememb'
    batch_size = 2048

    run_and_evaluate(
        dataset_path,
        args.base_model,
        args.experiment_name,
        item_catgories_fname,
        user_basline_fname,
        args.compute_baseline,
        args.per_user_metrics,
    )