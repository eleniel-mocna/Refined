import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd

from comparison.gather_metrics import gather_metrics
from random_forest.random_forest import generate_RFC_model, save_RFC_model


def extract_small_dataset(arffs: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a small dataset from the given arffs.
    @param arffs: List of arffs
    @return: Tuple of data and labels
    """
    data = np.concatenate(list(map(lambda x: x.drop("@@class@@", axis=1), arffs)))
    labels = np.concatenate(list(map(lambda x: x["@@class@@"] == b'1', arffs)))
    return data, labels


def evaluate_rfc(train_data, train_labels, test_data, test_labels):
    folder = "rfc"
    if not os.path.exists(folder):
        os.mkdir(folder)
    rfc = generate_RFC_model(train_data, train_labels)
    save_RFC_model(rfc, os.path.join(folder, "rfc_model.pckl"))
    metrics = gather_metrics(rfc, test_data, test_labels, os.path.join(folder, "rfc_metrics.pckl"))
    return rfc, metrics


def main():
    with open("chen11.pckl", "rb") as file:
        train_arffs = pickle.load(file)
    X_train, y_train = extract_small_dataset(train_arffs)
    with open("holo4k.pckl", "rb") as file:
        test_arffs = pickle.load(file)
    X_test, y_test = extract_small_dataset(test_arffs)
    evaluate_rfc(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
