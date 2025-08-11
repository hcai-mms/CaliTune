import os
import shutil

from calitune.calitune import run_calitune_from_config_file
from evaluation.run_and_evaluate import run_and_evaluate

"""
This helper script executes all the steps to run multiple experiments right after each other.
"""

DATASET = 'ml-100k'
BASE_MODEL_NAME = 'bpr-ml-100k-256.pth'

EXPERIMENTS = [
    ("debug1",    "configurations/calitune/CaliTune_config_ml-100k.yaml"),
    # ("combined_mode_0.05", "configurations/ml-100k/ambar/combined_mode_0.05.yaml"),
    # ("combined_mode_0.1",  "configurations/ml-100k/ambar/combined_mode_0.1.yaml"),
    # ("combined_mode_0.25", "configurations/ml-100k/ambar/combined_mode_0.25.yaml"),
    # ("combined_mode_0.5",  "configurations/ml-100k/ambar/combined_mode_0.5.yaml"),
    # ("combined_mode_0.75", "configurations/ml-100k/ambar/combined_mode_0.75.yaml"),
    # ("combined_mode_1",    "configurations/ml-100k/ambar/combined_mode_1.yaml"),
]

if __name__ == "__main__":
    main_dir = os.getcwd()

    dataset_path = os.path.join("data", DATASET)
    eval_main_path = os.path.join("evaluation", "data", DATASET)
    # Create main folder in evaluation
    os.makedirs(eval_main_path, exist_ok=True)
    # Copy over base model
    shutil.copy(
        os.path.join("saved", BASE_MODEL_NAME), os.path.join(eval_main_path, BASE_MODEL_NAME),
    )

    for idx, (experiment_name, config_path) in enumerate(EXPERIMENTS):

        print("===========================")
        print(f"Running experiment: {experiment_name}")
        print("===========================")

        # Clean up any previous experiment data
        shutil.rmtree(os.path.join("data", DATASET, "saved_models"), ignore_errors=True)

        """Call calitune.py with the current experiment name and config path"""
        run_calitune_from_config_file(config_path, DATASET)

        # Create the folder in evaluation
        os.makedirs(os.path.join(eval_main_path, experiment_name), exist_ok=True)

        # Copy necessary files into the evalation folder: Base weights, popularity, userbaseline
        shutil.copy(
            os.path.join(dataset_path, "base_weights.pth"),
            os.path.join(eval_main_path, "base_weights.pth"),
        )
        shutil.copy(
            os.path.join(dataset_path, f"{DATASET}.useremb"),
            os.path.join(eval_main_path, experiment_name, f"{DATASET}.useremb"),
        )
        shutil.copy(
            os.path.join(dataset_path, f"{DATASET}.userbaseline"),
            os.path.join(eval_main_path, experiment_name, f"{DATASET}.userbaseline"),
        )
        shutil.copy(
            os.path.join(dataset_path, f"{DATASET}.popularity"),
            os.path.join(eval_main_path, experiment_name, f"{DATASET}.popularity"),
        )

        # Move saved_models folder to evaluation
        shutil.rmtree(os.path.join(eval_main_path, experiment_name, "saved_models"), ignore_errors=True)
        shutil.move(
            os.path.join(dataset_path, "saved_models"),
            os.path.join(eval_main_path, experiment_name),
        )

        """Call evaluation script, with baseline computation for first experiment"""
        run_and_evaluate(
            eval_main_path,
            BASE_MODEL_NAME,
            experiment_name,
            f"{DATASET}.popularity",
            f"{DATASET}.userbaseline",
            compute_baseline=idx == 0,
            per_user_metrics=True,
        )

        # copy the resulting metrics into a more convenient location
        os.makedirs(os.path.join("metrics", DATASET), exist_ok=True)
        os.makedirs(os.path.join("per_user_metrics", DATASET), exist_ok=True)
        shutil.copy(
            os.path.join(eval_main_path, experiment_name, "saved_models", "result_saved_models_test.json"),
            os.path.join("metrics", DATASET, f"{experiment_name}.json"),
        )
        shutil.copy(
            os.path.join(eval_main_path, experiment_name, "saved_models",
                         "per_user_results_saved_models_test.tsv.zip"),
            os.path.join("per_user_metrics", DATASET, f"{experiment_name}.tsv.zip"),
        )