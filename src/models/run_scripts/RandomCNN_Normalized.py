import sys

import numpy as np

from config.config import Config
from models.evaluation.ModelEvaluator import ModelEvaluator
from models.refined.Random_refined import RandomRefinedNormalized
from models.refined.RefinedModel import generate_refined_model

np.set_printoptions(threshold=sys.maxsize)
import pickle

best_params = {"initial_size": 128,
               "growth_factor": 1.4,
               "last_dense": 128,
               "cnn_layers": 3,
               "cnn_stride": 1,
               "kernel_size": 3,
               "learning_rate": 5e-05,
               }


def main():
    with open(config.train_surroundings, "rb") as file:
        data, labels = pickle.load(file)

    refined = RandomRefinedNormalized(data, 38, 30)

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
