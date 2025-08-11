from recbole.quick_start import run_recbole
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='BPR')
    parser.add_argument('-d', '--dataset', type=str, default='ml-100k')
    parser.add_argument('-c', '--config_file_list', type=str, nargs='+',
                        default=[str(os.path.join("configurations", "recbole", "Recbole_BPR_config_ml-100k.yaml"))])
    args = parser.parse_args()

    run_recbole(model=args.model, dataset=args.dataset, config_file_list=args.config_file_list)
