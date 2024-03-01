import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

from config.config import Config
from models.common.ProteinModel import ProteinModel


class ModelEvaluator:
    def __init__(self, model: ProteinModel):
        self.model = model
        self.results = dict()

        self.data, self.labels = ModelEvaluator._get_data()

    @property
    def flat_data(self):
        return pd.concat(self.data[:2], ignore_index=True)

    @property
    def flat_labels(self):
        return pd.concat(self.labels[:2], ignore_index=True)

    def calculate_basic_metrics(self) -> 'ModelEvaluator':
        self.results["Accuracy"] = accuracy_score(self.flat_labels, self.model.predict(self.flat_data))
        self.results["Precision"] = precision_score(self.flat_labels, self.model.predict(self.flat_data),
                                                    average='macro')
        self.results["Recall"] = recall_score(self.flat_labels, self.model.predict(self.flat_data), average='macro')
        self.results["F1_score"] = f1_score(self.flat_labels, self.model.predict(self.flat_data), average='macro')
        return self

    def calculate_session_metrics(self) -> 'ModelEvaluator':
        accuracies = [accuracy_score(self.labels[i], self.model.predict(self.data[i])) for i in range(len(self.data))]
        self.results["Session Accuracy Min"] = min(accuracies)
        self.results["Session Accuracy Max"] = max(accuracies)
        self.results["Session Accuracy Var"] = np.var(accuracies)
        self.results["Session Accuracy Median"] = np.median(accuracies)
        return self

    def save_to_file(self, file_name: Path):
        tabulated = tabulate(self.results.items(), tablefmt="github", headers=["Metric", "Value"])
        with open(file_name, "w") as file:
            file.write(tabulated)

    @staticmethod
    def _get_data() -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        # TODO: Implement a data loader
        config = Config.get_instance()
        with open(config.test_extracted, "rb") as file:
            arffs = pickle.load(file)
        arffs = arffs[:30]
        print("Data loaded.")
        print("Preparing data...")
        labels = list(map(lambda x: x["@@class@@"] == b'1', arffs))
        data = list(map(lambda x: x.drop("@@class@@", axis=1), arffs))
        return data, labels
