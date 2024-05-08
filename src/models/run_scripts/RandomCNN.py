"""
This is a python script that trains a Random CNN model. It accepts a CLI argument:
--tune-hyperparameters: If specified, hyperparameter tuning will be done,
    otherwise previously found hyperparameters will be used.
"""
import sys

import numpy as np

from config.config import Config
from models.common.run_script_functions import train_surroundings_model
from models.refined.Random_refined import RandomRefined
from models.refined.RefinedModel import generate_refined_model

np.set_printoptions(threshold=sys.maxsize)

best_hyperparams = {
    "initial_size": 128,
    "growth_factor": 0.95,
    "last_dense": 32,
    "cnn_layers": 1,
    "cnn_stride": 2,
    "kernel_size": 3,
    "learning_rate": 5e-05
}


def train_random_cnn(data, labels, hyperparams):
    refined = RandomRefined(38, Config.default().surroundings_size)
    return generate_refined_model(np.array(data),
                           np.array(labels),
                           image_transformer=refined,
                           hyperparams=hyperparams,
                           name_suffix="_random")


def main(tune_hyperparameters: bool = False):
    train_surroundings_model(train_random_cnn,
                             hyperparams=None if tune_hyperparameters else best_hyperparams,
                             )


if __name__ == '__main__':
    main(tune_hyperparameters="--tune-hyperparameters" in sys.argv)
