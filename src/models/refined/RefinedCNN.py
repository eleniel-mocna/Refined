import sys

import numpy as np

from config.config import Config
from models.evaluation.ModelEvaluator import ModelEvaluator
from models.refined.Refined import Refined
from models.refined.RefinedModel import generate_refined_model

np.set_printoptions(threshold=sys.maxsize)
import pickle


def main():
    with open(config.train_surroundings, "rb") as file:
        data, labels = pickle.load(file)

    refined = Refined(data, 38, 30, "temp", hca_starts=1)
    refined.run()
    print(f"Refined order: \n{refined.best_individual}")
    print(f"Refined score: \n{refined.hof.best_fitness}")

    rfc_surrounding_model, _ = generate_refined_model(np.array(data), np.array(labels), refined)
    rfc_surrounding_model.save_model()
    return (ModelEvaluator(rfc_surrounding_model)
            .calculate_basic_metrics()
            .calculate_session_metrics()
            .save_to_file(rfc_surrounding_model.get_result_folder() / "metrics.txt")
            .print())


config = Config.get_instance()

if __name__ == '__main__':
    main()
