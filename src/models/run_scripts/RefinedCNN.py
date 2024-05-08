"""
This is a python script that trains a REFINED model. It accepts two CLI arguments:
--run-refined: If specified, a new REFINED image transformer will be trained, otherwise a pretrained one will be used
--tune-hyperparameters: If specified, hyperparameter tuning will be done,
    otherwise previously found hyperparameters will be used.
"""
import json
import sys
from typing import Any, Dict, Optional

import numpy as np

from config.constants import REFINED_ORDERS
from models.common.run_script_functions import train_surroundings_model
from models.refined.Refined import Refined
from models.refined.RefinedModel import generate_refined_model, RefinedModel

np.set_printoptions(threshold=sys.maxsize)

best_params = {"initial_size": 1024,
               "growth_factor": 1.05,
               "last_dense": 16,
               "cnn_layers": 1,
               "cnn_stride": 3,
               "kernel_size": 5,
               "learning_rate": 0.0005,
               }


def main(run_refined: bool = True, tune_hyperparameters: bool = False):
    train_surroundings_model(train_refined_model,
                             run_refined=run_refined,
                             hyperparameters=best_params if tune_hyperparameters else None)


def train_refined_model(data,
                        labels,
                        run_refined: bool,
                        hyperparameters: Optional[Dict[str, Any]]) -> RefinedModel:
    refined = Refined(data, 38, 30, "temp", hca_starts=1)
    if run_refined:
        refined.run()
    else:
        refined.from_pretrained(np.array(json.load(open(REFINED_ORDERS))[-1]["order"]))
    refined_model, _ = generate_refined_model(np.array(data),
                                              np.array(labels),
                                              refined,
                                              hyperparameters)
    return refined_model


if __name__ == '__main__':
    main(run_refined="--run-refined" in sys.argv,
         tune_hyperparameters="--tune-hyperparameters" in sys.argv)
