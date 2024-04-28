import sys

import numpy as np

from config.config import Config
from models.common.data_splitter import DataSplitter
from models.evaluation.ModelEvaluator import ModelEvaluator, booleanize
from models.refined.Random_refined import RandomRefined
from models.refined.RefinedModel import generate_refined_model

np.set_printoptions(threshold=sys.maxsize)
import pickle

best_hyperparams = {
    "initial_size": 128,
    "growth_factor": 0.95,
    "last_dense": 32,
    "cnn_layers": 1,
    "cnn_stride": 2,
    "kernel_size": 3,
    "learning_rate": 5e-05
}


def main():
    with open(config.train_surroundings, "rb") as file:
        data, labels = pickle.load(file)
    labels = np.vectorize(booleanize)(labels)
    splitter = DataSplitter(data, labels, config.train_lengths, 5)
    for i in range(5):
        data, labels = splitter.get_split(i)
        refined = RandomRefined(38, 30)

        rfc_surrounding_model, _ = generate_refined_model(np.array(data), np.array(labels), refined, best_hyperparams, "_random")
        rfc_surrounding_model.save_model()
        return (ModelEvaluator(rfc_surrounding_model)
                .calculate_basic_metrics()
                .calculate_session_metrics()
                .save_to_file(rfc_surrounding_model.get_result_folder() / "metrics.txt")
                .print())


config = Config.get_instance()

if __name__ == '__main__':
    main()
