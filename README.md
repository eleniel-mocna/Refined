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
will save the results in the `data` folder as described in the [Output section](#Outputs-and-data).

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

## Outputs and data

All data used in this pipeline is stored in the `data` folder. The following folders and files are included in there:

- `raw/`: Folder with the raw arffs, from which the data is extracted.
- `extracted/`: Folder with the loaded datasets saved as a pickles.
- `surroundings/`: Folder with created surroundings datasets saved as pickles.
- `models/`: Folder with evaluated models and files for their re-loading.
- `REFINED_values.json`: Json file with cached REFINED permutations, as they were created during a previous REFINED
  run (does not update)
- `images/`: Folder with images and visualizations for the thesis.
- `published_models/`: Folder with evaluations of pretrained models, shown as examples.

## Configuration

The configuration of the pipeline is done via the `config.json` file. In this file the following items can be set:

- `extract_dataset`: The dataset to be processed by data preprocessing steps. By default, "*" for processing all
  datasets.
- `train_dataset`: The dataset to be used as the training dataset. By default, "chen11".
- `test_dataset`: The dataset to be used at the testing dataset. By default, "coach420".
- `extraction_size`: Maximum limit for how many proteins should be read from arffs. Set to 0 for using all proteins.
- `surroundings_limit`: Maximum limit for how many proteins should the surroundings be calculated for. Set to 0 for
  using
  all proteins.
- `train_size`: Limit for how many proteins should be used for the dataset training. Most models use a part of
  this dataset as the validation dataset (for hyperparameter training etc.)
- `test_size`: Limit on how many proteins should be used for model testing.
- `surroundings_size`: The size of the surroundings for each SAS point. Default is 30. **If changed, remove
  the `surroundings` folder and rerun the computations**.
- `model_splits`: Number of "cross-validation" splits for model training. If set to 1, all data is used for 1 model. If
  set to a larger number _k_, _k_ models are trained, where each model is trained on _(k-1)/k_ part of the training
  dataset. For example, if _k=3_, 3 models are trained (A,B,C) and the train dataset is split into 3 parts of the same
  size (a,b,c) then model A is trained on dataset parts b, c; B - a,c; C - a,b.

# Adding new models and metrics

This codebase was made for running experiments for my bachelor thesis. Nonetheless, it can be used to validate other
methods on these datasets - or try new approaches for the existing models. Here, I will shortly describe the
architecture and some common edits.

## Data preprocessing

The data preprocessing has 3 stages. These can be all done by the `Run Whole Pipeline` run configuration or individual
scripts as described below.

1) **Reading the data**: This part of code is done by the `Extract arffs to pickle` run configuration. It is a simple
   script that reads all the data in the arff format, loads it as a pd.DataFrame and saves the whole list into
   the `data/extracted` folder.
2) **Extracting surroundings**: This part of code is done by the `Extract Surroundings` run configuration. It uses
   the `SurroundingsExtractor.extract_surroundings()` method to transform a list of pd.DataFrames and convert it into a
   np.array of data and a np.array of labels. This script also saves the protein lengths for evaluation purposes.

## Training the models

Each model has a script in the `src/models/run_scripts` folder. This is mostly run by the `Run Whole Pipeline` run
configuration, but can also be run separately.

Each model needs to implement the `ProteinModel` interface. For models using the surroundings
dataset, `SurroundingsProteinModel` interface is better, as it uses the data in the surroundings form. For each
surroundings model a method such as `train_refined_model` is implemented. This method
accepts `data: np.array`, `labels:np.array` and then any model-specific keyword arguments and returns an
implemented `SurroundingsProteinModel`. This method can be then passed to the `train_surroundings_model`, which then
trains the model on the configured data, does "cross-validation" reruns and evaluates the data.

## Evaluating the models

All model evaluation is taken care of by the `ModelEvaluator` class. It accepts a model and then can calculate and save
all the needed metrics. This class is called in the `train_surroundings_model`. It should be used the same way as there
in all uses.
