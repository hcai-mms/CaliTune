# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29, 2022/7/13, 2022/7/18
# @Author : Zihan Lin, Yupeng Hou, Gaowei Zhang, Lei Wang
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn, zgw15630559577@163.com, zxcptss@gmail.com

import argparse
import os
import numpy as np
import logging
from logging import getLogger

import ray
from ray import tune
from ray.tune.experiment import Trial
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Callback
import math

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)

from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
)

# Utility to report the best trials in Console during execution
class CustomTuneReporter(Callback):
    def __init__(self, metric="ndcg@10", mode="max", top_n=5):
        super().__init__()
        self.metric = metric
        self.mode = mode
        self.top_n = top_n
        self.best_trials = []

    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called every time a trial reports a result."""
        if len(self.best_trials) < self.top_n:
            self.best_trials.append((trial, result[self.metric]))
        else:
            # Find the worst performing trial in the current best trials
            worst_index = 0
            for i in range(1, len(self.best_trials)):
                if (self.mode == "max" and self.best_trials[i][1] < self.best_trials[worst_index][1]) or \
                        (self.mode == "min" and self.best_trials[i][1] > self.best_trials[worst_index][1]):
                    worst_index = i

            # Replace the worst trial if the current trial is better
            if (self.mode == "max" and result[self.metric] > self.best_trials[worst_index][1]) or \
                    (self.mode == "min" and result[self.metric] < self.best_trials[worst_index][1]):
                self.best_trials[worst_index] = (trial, result[self.metric])

        # Sort the best trials
        self.best_trials = sorted(self.best_trials, key=lambda x: x[1], reverse=(self.mode == "max"))

    def on_trial_complete(self, iteration, trials, trial, **info):
        self.print_status(trials)

    def on_trial_error(self, iteration, trials, trial, **info):
        self.print_status(trials)

    def on_experiment_end(self, trials, **info):
        self.print_status(trials)

    def print_status(self, trials):
        """Prints the current status of the Ray Tune run."""
        print("\n--- Current Best Results ---")

        # Running tasks
        running_trials = [t for t in trials if t.status == Trial.RUNNING]
        print(f"Running tasks: {len(running_trials)}")
        for trial in running_trials:
            print(f"  - {trial.trial_id}: {trial.config}")

        # Best tasks
        print(f"\nTop {self.top_n} Best tasks:")
        for trial, metric_value in self.best_trials:
            print(f"  - {trial.trial_id}: {metric_value} ({trial.config})")

        print("--- End Status ---\n")


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""(Copied from Recbole) The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    # As Item and User embedding sizes need to be compatible and there is no way to enforce this restriction
    # in the hyperparameters file, instead the
    DMF = True
    if DMF:
        config_dict['user_embedding_size'] = config_dict['embedding_size']
        config_dict['item_embedding_size'] = config_dict['embedding_size']
        del config_dict['embedding_size']

        config_dict['user_hidden_size_list'] = config_dict['hidden_size_list']
        config_dict['item_hidden_size_list'] = config_dict['hidden_size_list']
        del config_dict['hidden_size_list']

    # Instruct Recbole to use the GPU designated for this task
    config_dict['gpu_id'] = ray.get_gpu_ids()[0]

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return test_result


def ray_tune(args):
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    config_file_list = (
        [os.path.join(os.getcwd(), file) for file in config_file_list]
        if args.config_files
        else None
    )
    params_file = (
        os.path.join(os.getcwd(), args.params_file) if args.params_file else None
    )
    ray.init()
    tune.register_trainable("train_func", objective_function)
    config = {}
    with open(params_file, "r") as fp:
        for line in fp:
            para_list = line.strip().split(" ")
            if len(para_list) < 3:
                continue
            para_name, para_type, para_value = (
                para_list[0],
                para_list[1],
                "".join(para_list[2:]),
            )
            if para_type == "choice":
                para_value = eval(para_value)
                config[para_name] = tune.grid_search(para_value)
            elif para_type == "uniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.uniform(float(low), float(high))
            elif para_type == "quniform":
                low, high, q = para_value.strip().split(",")
                config[para_name] = tune.quniform(float(low), float(high), float(q))
            elif para_type == "loguniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.loguniform(
                    math.exp(float(low)), math.exp(float(high))
                )
            else:
                raise ValueError("Illegal param type [{}]".format(para_type))
    # choose different schedulers to use different tuning optimization algorithms
    # For details, please refer to Ray's official website https://docs.ray.io
    scheduler = ASHAScheduler(
        metric="ndcg@10", mode="max", max_t=10, grace_period=1, reduction_factor=2
    )

    local_dir = "./log_ray"

    custom_reporter = CustomTuneReporter(metric="ndcg@10", mode="max", top_n=5)

    result = tune.run(
        tune.with_parameters(objective_function, config_file_list=config_file_list),
        config=config,
        num_samples=1,
        log_to_file=args.output_file,
        scheduler=scheduler,
        local_dir=local_dir,
        resources_per_trial={"cpu": 1, "gpu": 1},
        callbacks=[custom_reporter]
    )

    best_trial = result.get_best_trial("ndcg@10", "max", "last")
    print("best params: ", best_trial.config)
    print("best result: ", best_trial.last_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_files", type=str, default=None, help="fixed config files"
    )
    parser.add_argument("--params_file", type=str, default=None, help="parameters file")
    parser.add_argument(
        "--output_file", type=str, default="hyper_example.result", help="output file"
    )
    args, _ = parser.parse_known_args()

    ray_tune(args)
