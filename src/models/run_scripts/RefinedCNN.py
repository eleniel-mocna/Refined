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

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data = data[indices]
    labels = labels[indices]
    refined = Refined(data, 38, 30, "temp", hca_starts=1)
    refined.from_pretrained(np.array(json.load(open(REFINED_ORDERS))[-1]["order"]))
    # refined.run()
    split_data, split_labels = np.array_split(data, 5), np.array_split(labels, 5)
    for i in range(5):
        print(f"Training RefinedCNN number {i}.")
        train_data = np.concatenate([split_data[j] for j in range(5) if j != i])
        train_labels = np.concatenate([split_labels[j] for j in range(5) if j != i])


        rfc_surrounding_model, _ = generate_refined_model(np.array(train_data), np.array(train_labels), refined, best_params)
        rfc_surrounding_model.save_model()
        (ModelEvaluator(rfc_surrounding_model)
         .calculate_basic_metrics()
         .calculate_session_metrics()
         .save_to_file(rfc_surrounding_model.get_result_folder() / "metrics.txt")
         .print())


if __name__ == '__main__':
    main()
