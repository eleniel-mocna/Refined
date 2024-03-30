import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config.config import Config
from models.big_rfc.RFCSurrounding import RFCSurrounding
from models.cutoff import get_best_cutoff
from models.evaluation.ModelEvaluator import ModelEvaluator, booleanize

np.set_printoptions(threshold=sys.maxsize)
import pickle
from sklearn.model_selection import train_test_split


def generate_BigRFC_model(data: np.ndarray, labels: np.ndarray) -> RFCSurrounding:
    """
    Create a GeneralModel implementation using a RandomForestClassifier.
    @param data: Numpy array of atom neighborhoods
    @param labels: Numpy array of labels for each atom
    @return: trained RandomForestModel
    """
    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels, test_size=0.20, random_state=42)
    y_train = train_labels
    y_test = test_labels

    X_train = train_data
    X_test = test_data
    print("Data prepared, training RFC...")
    random_forest: RandomForestClassifier = RandomForestClassifier(max_depth=5, n_jobs=15, verbose=3, n_estimators=15)
    random_forest.fit(X_train, y_train)
    print("RFC trained, finding best cutoff...")
    best_cutoff = get_best_cutoff(X_test, y_test, random_forest)
    print(best_cutoff)
    return RFCSurrounding(random_forest, best_cutoff)


def save_BigRFC_model(model: RFCSurrounding, file: str):
    """
    Save a RandomForestModel to a pickle file.
    @param model: RandomForestModel
    @param file: pickle file
    """
    with open(file, "wb") as file:
        pickle.dump(model, file)


def load_BigRFC_model(file: str) -> RFCSurrounding:
    """
    Load a RandomForestModel from a pickle file.
    @param file: pickle file
    @return: RandomForestModel
    """
    with open(file, "rb") as file:
        random_forest = pickle.load(file)
    return random_forest


def main():
    config = Config.get_instance()
    with open(config.train_surroundings, "rb") as file:
        data, labels = pickle.load(file)
    labels = np.vectorize(booleanize)(labels)

    rfc_surrounding_model = generate_BigRFC_model(np.array(data), np.array(labels))
    rfc_surrounding_model.save_model()
    (ModelEvaluator(rfc_surrounding_model)
     .calculate_basic_metrics()
     .calculate_session_metrics()
     .save_to_file(rfc_surrounding_model.get_result_folder() / "metrics.txt"))


if __name__ == '__main__':
    main()
