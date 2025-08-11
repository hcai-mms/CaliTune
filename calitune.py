import argparse
import os

from calitune.calitune import run_calitune_from_config_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config_path', type=str,
                        default=str(os.path.join("configurations", "calitune", "CaliTune_config_ml-100k.yaml")))
    parser.add_argument('-d', '--dataset', type=str, default="ml-100k")
    args = parser.parse_args()
    run_calitune_from_config_file(args.config_path, args.dataset)