# REFINED on P2Rank

This is a pipeline for training a REFINED model and comparing it to other models on P2Rank-based data.

# Running the code

This code should be runnable on all computers with Python 3.10+. In this README, I describe running the code on the
following setups:

- (Linux) machine with Bash (the main method)
- PyCharm based run configurations (can be unreliable and is not a CLI tool)

First, I'll describe the Bash way, then I'll describe the PyCharm way. After that, I will describe the output of the
runs and then custom configurable details.

## Dataset setup

Obtain the dataset from one of the following sources:

- Use the reduced dataset attached in the repository (no further action required, **BUT this is not the full dataset**)
- Copy the dataset attached to the bachelor thesis into the `data/raw` folder.
- Download the dataset from
  the [MFF storage](https://cunicz-my.sharepoint.com/personal/89562630_cuni_cz/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F89562630%5Fcuni%5Fcz%2FDocuments%2FDavid%5Farffs&ga=1)(
  access permissions required) into `data/raw`

## Running code via bash

### Setup

The dependencies for this project are governed by pip. Run the following commands in the root directory to create a
virtual environment, install all the required packages and set up the Python Path.

```shell
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install --no-cache-dir --upgrade pip setuptools
pip3 install --no-cache-dir -r requirements.txt
export export PYTHONPATH=$PYTHONPATH:src
```

### Running the code - 1 step

To run all the computations required, run the `run_configs/all.sh` script. This will run all the required scripts and
will save the results in the `data` folder as described in the [Output section](#Output).

### Running the code - multiple steps

As running the full pipeline takes multiple days, you can run the script in multiple steps:

1) Run `run_configs/bash_per_step/0_extract_arffs_to_pickle.sh` to read the dataset and save it as a pickle file.
2) Run `run_configs/bash_per_step/1_extract_surroundings.sh` to calculate the surroundings dataset. This step takes a
   considerable amount of time(On my AMD Ryzen 7 PRO 5850U, running on 16 threads, it runs 1-3 proteins per minute).
3) Run `run_configs/bash_per_step/2_train_models.sh` to train all the models described in the experiments section.

## PyCharm setup

- Set up a Pycharm environment as usual (if some issues emerge, use the Python CLI as described above and use the
  interpreter in the `./venv` environment).
- Run the `Run Whole Pipeline` configuration

Or this can be run in parts by running:

- `Extract arffs to pickle` to load the data from arff format.
- `Extract Surroundings` to extract surroundings dataset from the proteins.
- `Run Model Training` to conduct all the experiments.

## Running hyperparameter optimizations and other custom scripts

All of the scripts mentioned above do not run the hyperparameter optimization steps, as they take a long time.
Because of the way the repository is set up, before running any python scripts, adding the `src` to PythonPath is
needed.
To do this, run the following command

```shell
export export PYTHONPATH=$PYTHONPATH:src
```

After this is run, you can run the pipeline manually with the `src/models/run_scripts/whole_pipeline.py` script. It runs
the whole pipeline and train all the models based on the config file.
It recognizes the following arguments:

- `--skip-preprocessing`: If specified, only the models will be trained
- `--skip-model`-training: If specified, only data preprocessing will be done
- `--run-refined`: If specified, a new REFINED image transformer will be trained, otherwise a pretrained one will be
  used
- `--tune-hyperparameters`: If specified, hyperparameter tuning will be done,
  otherwise previously found hyperparameters will be used.

This can be used to retrain the REFINED model and run the hyperparameter tuning for all models.

For more precise running, you can also use all the run scripts in `src/models/run_scripts/`. These also accept the CLI
arguments for custom logic, but this is model specific, so I will not go into details with this.

# Output and data structure

All data used in this pipeline is stored in the `data` folder. The following folders and files are included in there:

- `raw/`: Folder with the raw arffs, from which the data is extracted.
- `extracted/`: Folder with the loaded datasets saved as a pickles.
- `surroundings/`: Folder with created surroundings datasets saved as pickles.
- `models/`: Folder with evaluated models and files for their re-loading.
- `REFINED_values.json`: Json file with cached REFINED permutations, as they were created during a previous REFINED
  run (does not update)
- `images/`: Folder with images and visualizations for the thesis.
- `published_models/`: Folder with evaluations of pretrained models, shown as examples.

