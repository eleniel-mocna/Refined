import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.core._exceptions import _ArrayMemoryError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

from config.config import Config
from models.common.ProteinModel import ProteinModel, SurroundingsProteinModel


class ModelEvaluator:
    def __init__(self, model: ProteinModel):
        self.model = model
        self.results = dict()
        self.config = Config.get_instance()

        self.is_surroundings_model = isinstance(model, SurroundingsProteinModel)
        if self.is_surroundings_model:
            model: SurroundingsProteinModel
            print("IS SMART!")
            self.data, self.labels = self._get_surroundings_data()
            try:
                self.y_pred = model.predict_surroundings(self.data)
            except MemoryError:
                # predict by batches
                print("MemoryError, predicting by batches")
                n = 10000
                self.y_pred = np.array([])
                for i in range(0, len(self.data), n):
                    print(f"Predicting batch {i//n}/{len(self.data)//n}")
                    self.y_pred = np.concatenate((self.y_pred, model.predict_surroundings(self.data[i:i+n])))

        else:
            print("Using basic data")
            self.data, self.labels = self._get_basic_data()
            self.y_pred = self.model.predict(self.flat_data)

    @property
    def flat_data(self):
        return self.data if self.is_surroundings_model else pd.concat(self.data, ignore_index=True)

    @property
    def flat_labels(self):
        return self.labels if self.is_surroundings_model else np.concatenate(self.labels)

    def split_by_protein(self, labels:np.array):
        proteins = [labels[i:j] for i, j in zip([0] + list(np.cumsum(self.get_proteins_lengths())[:-1]),
                                                np.cumsum(self.get_proteins_lengths()))]
        return proteins

    def get_proteins_lengths(self) -> List[int]:
        return self.config.test_lengths
    def calculate_basic_metrics(self) -> 'ModelEvaluator':
        self.results["N predictions"] = len(self.y_pred)
        self.results["Accuracy"] = accuracy_score(self.flat_labels, self.y_pred)
        self.results["Precision"] = precision_score(self.flat_labels, self.y_pred, average='macro')
        self.results["Recall"] = recall_score(self.flat_labels, self.y_pred, average='macro')
        self.results["F1_score"] = f1_score(self.flat_labels, self.y_pred, average='macro')
        return self

    def calculate_session_metrics(self) -> 'ModelEvaluator':
        labels = self.split_by_protein(self.flat_labels)
        y_pred = self.split_by_protein(self.y_pred)
        self.results["N Proteins"] = len(labels)
        accuracies = [accuracy_score(label, pred) for label, pred in zip(labels, y_pred) if label.shape[0]]
        confidence_interval = 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))

        self.results["Session Accuracy Median"] = np.median(accuracies)
        self.results["Session Accuracy Mean"] = np.mean(accuracies)
        self.results["Session Accuracy CI min"] = np.mean(accuracies) - confidence_interval
        self.results["Session Accuracy CI max"] = np.mean(accuracies) + confidence_interval

        precisions = [precision_score(label, pred) for label, pred in zip(labels, y_pred) if label.shape[0]]
        confidence_interval = 1.96 * np.std(precisions) / np.sqrt(len(precisions))
        self.results["Session Precision Median"] = np.median(precisions)
        self.results["Session Precision Mean"] = np.mean(precisions)
        self.results["Session Precision CI min"] = np.mean(precisions) - confidence_interval
        self.results["Session Precision CI max"] = np.mean(precisions) + confidence_interval

        recalls = [recall_score(label, pred) for label, pred in zip(labels, y_pred) if label.shape[0]]
        confidence_interval = 1.96 * np.std(recalls) / np.sqrt(len(recalls))
        self.results["Session Recall Median"] = np.median(recalls)
        self.results["Session Recall Mean"] = np.mean(recalls)
        self.results["Session Recall CI min"] = np.mean(recalls) - confidence_interval
        self.results["Session Recall CI max"] = np.mean(recalls) + confidence_interval

        f1_scores = [f1_score(label, pred) for label, pred in zip(labels, y_pred) if label.shape[0]]
        confidence_interval = 1.96 * np.std(f1_scores) / np.sqrt(len(f1_scores))
        self.results["Session F1_score Median"] = np.median(f1_scores)
        self.results["Session F1_score Mean"] = np.mean(f1_scores)
        self.results["Session F1_score CI min"] = np.mean(f1_scores) - confidence_interval
        self.results["Session F1_score CI max"] = np.mean(f1_scores) + confidence_interval

        return self

    def save_to_file(self, file_name: Path):
        tabulated = tabulate(self.results.items(), tablefmt="github", headers=["Metric", "Value"])
        with open(file_name, "w") as file:
            file.write(tabulated)
        json_file = file_name.with_suffix(".json")
        with open(json_file, "w") as file:
            json.dump(self.results, file, indent=4)
        return self

    def print(self):
        print(tabulate(self.results.items(), tablefmt="github", headers=["Metric", "Value"]))
        return self

    def _get_basic_data(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        # TODO: Implement a data loader
        with open(self.config.test_extracted, "rb") as file:
            arffs = pickle.load(file)
        print("Data loaded.")
        print("Preparing data...")
        labels = list(map(lambda x: np.vectorize(booleanize)(x["@@class@@"].to_numpy()), arffs))
        data = list(map(lambda x: x.drop("@@class@@", axis=1), arffs))
        return data, labels

    def _get_surroundings_data(self) -> Tuple[np.array, np.array]:
        data: np.array
        labels: np.array
        with open(self.config.test_surroundings, "rb") as file:
            data, labels = pickle.load(file)
        labels = np.vectorize(booleanize)(labels)
        return data, labels


def booleanize(x: np.array):
    if x in (True, 1, b'1', b'True'):
        return True
    if x in (False, 0, b'0', b'False'):
        return False
    raise ValueError
