import json
import sys

import numpy as np

from config.config import Config
from config.constants import REFINED_ORDERS
from models.common.data_splitter import DataSplitter
from models.evaluation.ModelEvaluator import ModelEvaluator, booleanize
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
    with open(config.train_surroundings, "rb") as file:
        data, labels = pickle.load(file)
    labels = np.vectorize(booleanize)(labels)
    splitter = DataSplitter(data, labels, config.train_lengths, 5)
    for i in range(5):
        data, labels = splitter.get_split(i)
        refined = Refined(data, 38, 30, "temp", hca_starts=1)
        refined.from_pretrained(np.array(json.load(open(REFINED_ORDERS))[-1]["order"]))

        rfc_surrounding_model, _ = generate_refined_model(np.array(data), np.array(labels), refined, best_params)
        rfc_surrounding_model.save_model()
        return (ModelEvaluator(rfc_surrounding_model)
                .calculate_basic_metrics()
                .calculate_session_metrics()
                .save_to_file(rfc_surrounding_model.get_result_folder() / "metrics.txt")
                .print())


config = Config.get_instance()

if __name__ == '__main__':
    main()
