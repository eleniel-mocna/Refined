import os
import pickle
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from rfc_baseline.RandomForestModel import RandomForestModel


def main():
    print("Loading data...")
    folder = "random_forest"
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_file = os.path.join(folder, "metrics.txt")
    with open("allArffs.pckl", "rb") as file:
        arffs = pickle.load(file)
    print("Data loaded.")

    print("Preparing data...")
    labels = list(map(lambda x: x["@@class@@"] == b'1', arffs))
    data = list(map(lambda x: x.drop("@@class@@", axis=1), arffs))
    random_forest = generate_RFC_model(data, labels)
    pickle_file_name = os.path.join(folder, "RFC.pckl")
    print(f"Saving RFC to {pickle_file_name}")
    save_RFC_model(random_forest, pickle_file_name)


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
    y_train = train_labels
    y_test = test_labels

    X_train = train_data
    X_test = test_data
    print("Data prepared, training RFC...")
    random_forest: RandomForestClassifier = RandomForestClassifier(max_depth=5, n_jobs=15, verbose=3, n_estimators=100)
    random_forest.fit(X_train, y_train)
    print("RFC trained, finding best cutoff...")
    best_cutoff = get_best_cutoff(X_test, y_test, random_forest)
    print(best_cutoff)
    return RandomForestModel(random_forest, best_cutoff)


def save_RFC_model(model: RandomForestModel, file: str):
    """
    Save a RandomForestModel to a pickle file.
    @param model: RandomForestModel
    @param file: pickle file
    """
    with open(file, "wb") as file:
        pickle.dump(model, file)


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
