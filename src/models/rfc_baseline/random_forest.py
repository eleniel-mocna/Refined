import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config.config import Config
from models.evaluation.ModelEvaluator import ModelEvaluator
from models.rfc_baseline.random_forest_model import RandomForestModel


def main():
    config = Config.get_instance()
    with open(config.train_extracted, "rb") as file:
        arffs = pickle.load(file)
    print("Data loaded.")

    print("Preparing data...")
    labels = list(map(lambda x: x["@@class@@"] == b'1', arffs))
    data = list(map(lambda x: x.drop("@@class@@", axis=1), arffs))
    random_forest = generate_RFC_model(data, labels)
    random_forest.save_model()
    (ModelEvaluator(random_forest)
     .calculate_basic_metrics()
     .calculate_session_metrics()
     .save_to_file(random_forest.get_result_folder() / "metrics.txt"))


def get_best_cutoff(data, labels, random_forest):
    y_pred = random_forest.predict_proba(data)
    best_cutoff = 0
    best_f1 = 0
    for i in range(1, 100):
        cutoff = i / 100
        f1 = f1_score(labels, y_pred[:, 1] > cutoff)
        if f1 > best_f1:
            best_cutoff = cutoff
            best_f1 = f1
    print(f"RFC trained with f1: {best_f1}")
    return best_cutoff


def generate_RFC_model(data: List[np.ndarray], labels: List[np.ndarray]) -> RandomForestModel:
    """
    Create a GeneralModel implementation using a RandomForestClassifier.
    @param data: List of proteins
    @param labels: List of labels for each atom in the protein
    @return: trained RandomForestModel
    """
    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels, test_size=0.20, random_state=42)
    y_train = pd.concat(train_labels)
    y_test = pd.concat(test_labels)

    X_train = pd.concat(train_data)
    X_test = pd.concat(test_data)
    print("Data prepared, training RFC...")
    random_forest: RandomForestClassifier = RandomForestClassifier(max_depth=None, n_jobs=15, verbose=3,
                                                                   n_estimators=200, max_features=6)
    random_forest.fit(X_train, y_train)
    print("RFC trained, finding best cutoff...")
    best_cutoff = get_best_cutoff(X_test, y_test, random_forest)
    print(best_cutoff)
    return RandomForestModel(random_forest, best_cutoff)


def load_RFC_model(file: str) -> RandomForestModel:
    """
    Load a RandomForestModel from a pickle file.
    @param file: pickle file
    @return: RandomForestModel
    """
    with open(file, "rb") as file:
        random_forest = pickle.load(file)
    return random_forest


if __name__ == '__main__':
    main()
