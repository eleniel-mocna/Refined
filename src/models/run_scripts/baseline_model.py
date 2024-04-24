import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config.config import Config
from models.common.ProteinModel import ProteinModel
from models.evaluation.ModelEvaluator import ModelEvaluator


def main():
    config = Config.get_instance()
    with open(config.train_extracted, "rb") as file:
        arffs = pickle.load(file)
    print("Data loaded.")

    print("Preparing data...")
    labels = list(map(lambda x: x["@@class@@"] == b'1', arffs))
    data = list(map(lambda x: x.drop("@@class@@", axis=1), arffs))
    for i in range(5):
        train_data, _,  train_labels, _ = train_test_split(data, labels, test_size=0.20, random_state=42)
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

    def __init__(self, random_forest: RandomForestClassifier):
        self.random_forest: RandomForestClassifier = random_forest

    def predict(self, protein: np.ndarray) -> np.ndarray:
        return self.random_forest.predict_proba(protein)[:, 1] > 0.5

    @staticmethod
    def from_data(data: List[pd.DataFrame], labels: List[pd.DataFrame]) -> 'RandomForestModel':
        """
        Create a model from REFINED original paper. From raw protein data.

        @param data: List of proteins
        @param labels: List of labels for each atom in the protein
        @return: trained RandomForestModel
        """
        train_labels_combined = pd.concat(labels)

        train_data_combined = pd.concat(data)
        print("Data prepared, training RFC...")
        random_forest: RandomForestClassifier = RandomForestClassifier(max_depth=None,
                                                                       n_jobs=-1,
                                                                       verbose=0,
                                                                       n_estimators=200,
                                                                       max_features=6)
        random_forest.fit(train_data_combined, train_labels_combined)
        return RandomForestModel(random_forest)


if __name__ == '__main__':
    main()
