import os
import json


def create_configs(base_path, experiment_name, item_catgories_fname, user_basline_fname):
    config = {
        "item_popularity": f"{base_path}/{experiment_name}/{item_catgories_fname}",
        "popularity_target": f"{base_path}/{experiment_name}/{user_basline_fname}",
        "base_recommendations": f"{base_path}/base_recs_test.npy",
        "hist_train": f"{base_path}/train_hist.npz",
        "hist_valid": f"{base_path}/valid_hist.npz",
        "hist_test": f"{base_path}/test_hist.npz",
        "postprocessed": {
            "0.1": f"{base_path}/{experiment_name}/postprocessed/result_0.1_test.npy",
            "0.2": f"{base_path}/{experiment_name}/postprocessed/result_0.2_test.npy",
            "0.3": f"{base_path}/{experiment_name}/postprocessed/result_0.3_test.npy",
            "0.4": f"{base_path}/{experiment_name}/postprocessed/result_0.4_test.npy",
            "0.5": f"{base_path}/{experiment_name}/postprocessed/result_0.5_test.npy",
            "0.6": f"{base_path}/{experiment_name}/postprocessed/result_0.6_test.npy",
            "0.7": f"{base_path}/{experiment_name}/postprocessed/result_0.7_test.npy",
            "0.8": f"{base_path}/{experiment_name}/postprocessed/result_0.8_test.npy",
            "0.9": f"{base_path}/{experiment_name}/postprocessed/result_0.9_test.npy",
        },
    }

    finetuned_base_path = f"{base_path}/{experiment_name}/"

    for parent in [
        f
        for f in os.listdir(finetuned_base_path)
        if os.path.isdir(os.path.join(finetuned_base_path, f))
    ]:
        finetuned_path = finetuned_base_path + parent + "/"
        curr_config = config.copy()
        curr_config["finetuned"] = {}
        folders = [
            f
            for f in os.listdir(finetuned_path)
            if os.path.isdir(os.path.join(finetuned_path, f))
        ]
        folders.sort(key=lambda x: int(x[6:]))
        for folder in folders:
            curr_config["finetuned"][
                folder
            ] = f"{base_path}/{experiment_name}/{parent}/{folder}/preds_finetuned_test.npy"
            with open(
                f"{base_path}/{experiment_name}/{parent}/test.config.json", "w"
            ) as f:
                json.dump(curr_config, f)


if __name__ == "__main__":
    base_path = "data/ML-1M_tuned_ep_all/experiments"
    experiment_name = "ADDITIONAL_POP"
    create_configs(base_path, experiment_name)
