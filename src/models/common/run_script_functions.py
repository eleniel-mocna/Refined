import pickle
from typing import Tuple

import numpy as np

from config.config import Config
from models.common.data_splitter import DataSplitter
from models.evaluation.ModelEvaluator import booleanize, ModelEvaluator


def get_train_surroundings() -> Tuple[np.array, np.array]:
    config = Config.get_instance()
    try:
        with open(config.train_surroundings, "rb") as file:
            data, labels = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Surroundings for the train dataset have not been  created.")
    labels = np.vectorize(booleanize)(labels)
    return data, labels


def evaluate_model(model):
    model.save_model()
    (ModelEvaluator(model)
     .calculate_basic_metrics()
     .calculate_session_metrics()
     .save_to_file(model.get_result_folder() / "metrics.txt")
     .print())


def train_surroundings_model(train_function, **kwargs):
    config = Config.get_instance()
    data, labels = get_train_surroundings()
    logger_string = (f"{train_function.__name__}:\n"
                     f"  data shape:            :\t{data.shape}\n"
                     f"  cross-validation splits:\t{config.model_splits}\n"
                     f"  with parameters:\n")
    logger_string += "\n".join([f"    {k}:\t{v}" for k, v in kwargs.items()])
    print(logger_string)

    splitter = DataSplitter(data, labels, config.train_lengths, config.model_splits)
    for i in range(config.model_splits):
        data, labels = splitter.get_split(i)
        model = train_function(data,
                               labels,
                               **kwargs)
        evaluate_model(model)
