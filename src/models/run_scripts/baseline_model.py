import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config.config import Config
from models.common.ProteinModel import ProteinModel
from models.common.cutoff import get_best_cutoff
from models.evaluation.ModelEvaluator import ModelEvaluator


def main():
    config = Config.get_instance()
    with open(config.train_extracted, "rb") as file:
        arffs = pickle.load(file)
    print("Data loaded.")

    print("Preparing data...")
    labels: List[pd.Series] = list(map(lambda x: x["@@class@@"] == b'1', arffs))
    data: List[pd.DataFrame] = list(map(lambda x: x.drop("@@class@@", axis=1), arffs))
    data, labels = shuffle(data, labels, random_state=42)
    split_data = [[data[j] for j in range(len(data)) if j % 5 == i] for i in range(5)]
    split_labels = [[labels[j] for j in range(len(labels)) if j % 5 == i] for i in range(5)]

    for i in range(5):
        print(f"Training model number {i}")
        # use all data except for i-th split
        train_data = [split_data[j] for j in range(5) if j != i]
        train_labels = [split_labels[j] for j in range(5) if j != i]

        # flatten the list of lists
        train_data = [item for sublist in train_data for item in sublist]
        train_labels = [item for sublist in train_labels for item in sublist]

        random_forest = RandomForestModel.from_data(train_data, train_labels)
        random_forest.save_model()
        (ModelEvaluator(random_forest)
         .calculate_basic_metrics()
         .calculate_session_metrics()
         .save_to_file(random_forest.get_result_folder() / "metrics.txt"))


class RandomForestModel(ProteinModel):
    def predict_proba(self, protein: pd.DataFrame) -> np.ndarray:
        return self.random_forest.predict_proba(protein)[:, 1]

    @property
    def name(self) -> str:
        return "RFC_baseline"

    def __init__(self, random_forest: RandomForestClassifier, cutoff: float):
        self.cutoff: float = cutoff
        self.random_forest: RandomForestClassifier = random_forest

    def predict(self, protein: np.ndarray) -> np.ndarray:
        return self.random_forest.predict_proba(protein)[:, 1] > self.cutoff

    @staticmethod
    def from_data(data: List[pd.DataFrame], labels: List[pd.Series]) -> 'RandomForestModel':
        """
        Create a model from REFINED original paper. From raw protein data.

        @param data: List of proteins
        @param labels: List of labels for each atom in the protein
        @return: trained RandomForestModel
        """
        train_data, test_data, train_labels, test_labels = \
            train_test_split(data, labels, test_size=0.20, random_state=42)
        train_labels_combined = pd.concat(train_labels)
        test_labels_combined = pd.concat(test_labels)

        train_data_combined = pd.concat(train_data)
        test_data_combined = pd.concat(test_data)
        print("Data prepared, training RFC...")
        random_forest: RandomForestClassifier = RandomForestClassifier(max_depth=None,
                                                                       n_jobs=-1,
                                                                       verbose=0,
                                                                       n_estimators=200,
                                                                       max_features=6)
        random_forest.fit(train_data_combined, train_labels_combined)
        print("RFC trained, finding best cutoff...")
        best_cutoff = get_best_cutoff(test_data_combined, test_labels_combined, random_forest)
        print(best_cutoff)
        return RandomForestModel(random_forest, best_cutoff)


if __name__ == '__main__':
    main()
