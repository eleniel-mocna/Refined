"""
This is a python script that trains a Normalized CNN model. It accepts a CLI argument:
--tune-hyperparameters: If specified, hyperparameter tuning will be done,
    otherwise previously found hyperparameters will be used.
"""
import sys

import numpy as np

from config.config import Config
from models.common.run_script_functions import train_surroundings_model
from models.refined.Random_refined import RandomRefinedNormalized
from models.refined.RefinedModel import generate_refined_model

np.set_printoptions(threshold=sys.maxsize)

best_params = {"initial_size": 128,
               "growth_factor": 1.4,
               "last_dense": 128,
               "cnn_layers": 3,
               "cnn_stride": 1,
               "kernel_size": 3,
               "learning_rate": 5e-05,
               }


def train_normalized_cnn(data, labels, hyperparams):
    refined = RandomRefinedNormalized(data, 38, Config.default().surroundings_size)
    return generate_refined_model(np.array(data),
                           np.array(labels),
                           image_transformer=refined,
                           hyperparams=hyperparams,
                           name_suffix="_random_norm")


def main(tune_hyperparameters: bool = False):
    train_surroundings_model(train_normalized_cnn,
                             hyperparams=None if tune_hyperparameters else best_params,
                             )


if __name__ == '__main__':
    main(tune_hyperparameters="--tune-hyperparameters" in sys.argv)
