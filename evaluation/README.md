# nDCG to JSD trade-off evaluation for Calibrated Popularity and CaliTune

# Description
The code in this folder, given a base model, runs Calibrated Popularity post-processing on it, and, given saved model states from CaliTune, evaluates both performances for nDCG@10 and JSD@10.
The result is `result_*_test.json`, containing the evaluation data for both approaches.

# Expected folder structure for evaluation
Place the base folder with the base model (from RecBole), fine-tuned model states, and popularity data (produced by CaliTune) next to `run_and_evaluate.py` and run it (make sure to `cd post-processing_and_evaluation` before running).
```
.
├─── base folder (eg. "data/base_model_1")
│   ├── experiment folder (eg. "experiment_1")
│   │   ├── folder with a CaliTune run ("saved_models")
│   │   │   └── folders for each epoch with saved CaliTune embeddings
│   │   ├── precalculated item popularity baseline file (eg. "ml-100k.popularity")
│   │   └── precalculated user popularity baseline file (eg. "ml-100k.userbaseline")
│   └── base model .pth file (eg. "ml-100k_base.pth")
├─── run_and_evaluate.py (pipeline launcher: CP and evaluation)
...
```

In practice the `saved_models` (eg. for Quick Start `data/ml-100k/saved_models`) folder from a CaliTune run needs to be copied to the level of "folder with a Coder run". Furthermore the base model (in the top level `saved` folder), interaction file and popularity files need to be copied from the paths specified in the Coder files to the evaluation experiment folder according to the specified structure. 


# Usage
Executing `run_and_evaluate.py` runs the postprocessing baseline and evaluates it together with a CaliTune run. Parameters (files and folder names as described in the previous section) need to be specified within the python script. Parameters for the postprocessing are hardcoded within the "main_coder.py" file.

The `run_and_evaluate.py` file is prepared to evaluate a ml-100k run. For a Quick Start copy the files as described in the "Expected folder structure for evaluation" section. Use the self trained model from the `saved` folder on the top level and rename it to `ml-100k_base.pth`.

The evaluation results `result_*_test.json` are found in the `folder with a CaliTune run` folder (eg. `data/base_model_1/experiment_1/saved_models`). All evaluations are only done on the test set.
