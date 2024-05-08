"""
This is a python script that trains a REFINED CNN on each progression of the pretrained REFINED core permutations.
"""
import json
import sys

import numpy as np

from config.config import Config
from config.constants import REFINED_ORDERS
from models.evaluation.ModelEvaluator import ModelEvaluator
from models.refined.Refined import Refined
from models.refined.RefinedModel import generate_refined_model

np.set_printoptions(threshold=sys.maxsize)
import pickle

best_params = {"initial_size": 1024,
               "growth_factor": 1.05,
               "last_dense": 16,
               "cnn_layers": 1,
               "cnn_stride": 3,
               "kernel_size": 5,
               "learning_rate": 0.0005,
               }


def main():
    config = Config.get_instance()
    with open(config.train_surroundings, "rb") as file:
        data, labels = pickle.load(file)

    refined_orders = json.load(open(REFINED_ORDERS))
    for i in range(len(refined_orders)):
        print(f"Running REFINED on evolution {i}")
        refined = Refined(data, 38, config.surroundings_size, "temp", hca_starts=1)
        refined.from_pretrained(np.array(refined_orders[-1]["order"]))

        rfc_surrounding_model = generate_refined_model(
            np.array(data),
            np.array(labels),
            refined,
            best_params,
            f"_{i}")
        rfc_surrounding_model.save_model()
        (ModelEvaluator(rfc_surrounding_model)
         .calculate_basic_metrics()
         .calculate_session_metrics()
         .save_to_file(rfc_surrounding_model.get_result_folder() / "metrics.txt")
         .print())



if __name__ == '__main__':
    main()
