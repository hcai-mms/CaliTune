import os
import shutil
from itertools import product

from evaluation.run_and_evaluate import run_and_evaluate
from calitune.calitune import run_calitune
from calitune.utils import read_config

DATASET = 'ml-1m-dmf'
BASE_MODEL_NAME = 'dmf-ml-1m-256-256.pth'
BASE_CONFIG_FILE = 'configurations/calitune/CaliTune_config_ml-1m-dmf.yaml'

COMBINED_MODE_ALPHA = [
    0.0,
    # 0.1,
    # 0.25,
    0.5,
    # 0.75,
    1.0
]
FROZEN_ITEM_EMB = [
    True,
    False
]
CANDIDATE_SELECTION_STRATEGIES = [
    'relevance',
    # 'random'
]
K_CANDIDATES = [
    #25,
    # 50,
    # 75,
    100,
    # 150,
    # 200,
    # 400
]
LEARNING_RATE = [
    # 0.001,
    0.0005,
    0.0001,
    0.00001,
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
    # Copy over base weights
    shutil.copy(
        os.path.join(dataset_path, "base_weights.pth"), os.path.join(eval_main_path, "base_weights.pth"),
    )



    # Run a grid search on all combinations of the parameters
    for idx, parameters in enumerate(product(
            COMBINED_MODE_ALPHA,
            FROZEN_ITEM_EMB,
            CANDIDATE_SELECTION_STRATEGIES,
            K_CANDIDATES,
            LEARNING_RATE
    )):
        combined_mode_alpha, frozen_item_emb, candidate_selection_strategy, k_candidates, learning_rate,  = parameters

        frozen_str = "frozen_itememb" if frozen_item_emb else "unfrozen_itememb"

        experiment_name = f'c{combined_mode_alpha}__{frozen_str}__selection_{candidate_selection_strategy}__k{k_candidates}__lr{learning_rate}'

        print("===========================")
        print(f"Running experiment: {experiment_name}")
        print("===========================")

        # Clean up any previous experiment data
        shutil.rmtree(os.path.join("data", DATASET, "saved_models"), ignore_errors=True)

        # Load config file
        config = read_config(BASE_CONFIG_FILE)
        # Update config parameters
        config['freeze_item_emb'] = frozen_item_emb
        config['candidate_selection'] = candidate_selection_strategy
        config['learning_rate'] = learning_rate
        config['combined_mode_alpha'] = combined_mode_alpha
        config['k_candidates'] = k_candidates

        """Call calitune.py with the current experiment name and config path"""
        run_calitune(config, DATASET)

        # Create the folder in evaluation
        os.makedirs(os.path.join(eval_main_path, experiment_name), exist_ok=True)

        # Copy necessary files into the evalation folder: Embeddings, popularity, userbaseline
        shutil.copy(
            os.path.join(dataset_path, f"{DATASET}.itememb"),
            os.path.join(eval_main_path, experiment_name, f"{DATASET}.itememb"),
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