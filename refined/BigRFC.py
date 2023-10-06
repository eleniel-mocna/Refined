import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from refined.RFCSurrounding import RFCSurrounding

np.set_printoptions(threshold=sys.maxsize)
import pickle
from sklearn.model_selection import train_test_split


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
    random_forest: RandomForestClassifier = RandomForestClassifier(max_depth=5, n_jobs=15, verbose=3, n_estimators=100)
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
    print("Loading data...")
    folder = "refined/RFC"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open("surroundings_dataset_1700.pckl", "rb") as file:
        data, labels = pickle.load(file)
    rfc_surrounding_model = generate_BigRFC_model(np.array(data), np.array(labels))
    with open(os.path.join(folder, "RFCSurrounding.pckl"), "wb") as file:
        pickle.dump(rfc_surrounding_model, file)


if __name__ == '__main__':
    main()
