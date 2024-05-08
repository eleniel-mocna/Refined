import copy
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from tabulate import tabulate

from config.config import Config
from models.common.ProteinModel import ProteinModel, SurroundingsProteinModel


class ModelEvaluator:
    def __init__(self, model: ProteinModel):
        self.model = model
        self.results = {"model_name": model.name}
        self.config = Config.get_instance()

        self.is_surroundings_model = isinstance(model, SurroundingsProteinModel)
        if self.is_surroundings_model:
            model: SurroundingsProteinModel
            print("IS SMART!")
            self.data, self.labels = self._get_surroundings_data()
            try:
                self.y_pred_prob = model.predict_surroundings_proba(self.data)
            except MemoryError:
                # predict by batches
                print("MemoryError, predicting by batches")
                n = 10000
                self.y_pred_prob = np.array([])
                for i in range(0, len(self.data), n):
                    print(f"Predicting batch {i // n}/{len(self.data) // n}")
                    self.y_pred_prob = np.concatenate(
                        (self.y_pred_prob, model.predict_surroundings_proba(self.data[i:i + n])))

        else:
            print("Using basic data")
            self.data, self.labels = self._get_basic_data()
            self.y_pred_prob = self.model.predict_proba(self.flat_data)

        # calculate the best threshold
        # Calculate the ROC
        self.fpr: np.ndarray
        self.tpr: np.ndarray
        self.thresholds: np.ndarray
        self.fpr, self.tpr, self.thresholds = roc_curve(self.flat_labels, self.y_pred_prob)

        # Get the best threshold for f1
        best_f1 = 0
        best_threshold = 0
        for i in range(len(self.fpr)):
            f1 = 2 * self.tpr[i] * (1 - self.fpr[i]) / (self.tpr[i] + 1 - self.fpr[i])
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = self.thresholds[i]
        self.y_pred = self.y_pred_prob > best_threshold

    @property
    def flat_data(self):
        return self.data if self.is_surroundings_model else pd.concat(self.data, ignore_index=True)

    @property
    def flat_labels(self):
        return self.labels if self.is_surroundings_model else np.concatenate(self.labels)

    def split_by_protein(self, labels: np.array):
        proteins = [labels[i:j] for i, j in zip([0] + list(np.cumsum(self.get_proteins_lengths())[:-1]),
                                                np.cumsum(self.get_proteins_lengths()))]
        return proteins

    def get_proteins_lengths(self) -> List[int]:
        return self.config.test_lengths

    def calculate_basic_metrics(self) -> 'ModelEvaluator':
        self.results["N predictions"] = len(self.y_pred)
        self.results["Accuracy"] = accuracy_score(self.flat_labels, self.y_pred)
        self.results["Accuracy CI max"] = accuracy_score(self.flat_labels, self.y_pred) + 1.96 * np.std(
            self.flat_labels == self.y_pred) / np.sqrt(len(self.y_pred))
        self.results["Accuracy CI min"] = accuracy_score(self.flat_labels, self.y_pred) - 1.96 * np.std(
            self.flat_labels == self.y_pred) / np.sqrt(len(self.y_pred))
        self.results["Precision"] = precision_score(self.flat_labels, self.y_pred, average='macro')
        self.results["Precision CI max"] = precision_score(self.flat_labels, self.y_pred,
                                                           average='macro') + 1.96 * np.std(
            self.flat_labels == self.y_pred) / np.sqrt(len(self.y_pred))
        self.results["Precision CI min"] = precision_score(self.flat_labels, self.y_pred,
                                                           average='macro') - 1.96 * np.std(
            self.flat_labels == self.y_pred) / np.sqrt(len(self.y_pred))
        self.results["Recall"] = recall_score(self.flat_labels, self.y_pred, average='macro')
        self.results["Recall CI max"] = recall_score(self.flat_labels, self.y_pred, average='macro') + 1.96 * np.std(
            self.flat_labels == self.y_pred) / np.sqrt(len(self.y_pred))
        self.results["Recall CI min"] = recall_score(self.flat_labels, self.y_pred, average='macro') - 1.96 * np.std(
            self.flat_labels == self.y_pred) / np.sqrt(len(self.y_pred))
        self.results["F1_score"] = f1_score(self.flat_labels, self.y_pred, average='macro')
        self.results["F1_score CI max"] = f1_score(self.flat_labels, self.y_pred, average='macro') + 1.96 * np.std(
            self.flat_labels == self.y_pred) / np.sqrt(len(self.y_pred))
        self.results["F1_score CI min"] = f1_score(self.flat_labels, self.y_pred, average='macro') - 1.96 * np.std(
            self.flat_labels == self.y_pred) / np.sqrt(len(self.y_pred))

        # AUC
        self.results["AUC"] = np.trapz(self.tpr, self.fpr)

        self.results["TP"] = int(np.sum(np.logical_and(self.flat_labels, self.y_pred)))
        self.results["FP"] = int(np.sum(np.logical_and(np.logical_not(self.flat_labels), self.y_pred)))
        self.results["TN"] = int(np.sum(np.logical_and(np.logical_not(self.flat_labels), np.logical_not(self.y_pred))))
        self.results["FN"] = int(np.sum(np.logical_and(self.flat_labels, np.logical_not(self.y_pred))))
        return self

    def calculate_session_metrics(self) -> 'ModelEvaluator':
        labels = self.split_by_protein(self.flat_labels)
        y_pred = self.split_by_protein(self.y_pred)
        self.results["N Proteins"] = len(labels)

        # Filter proteins with no predicted LBS
        true_proteins_without_lbs = np.array([max(label) == 0 for label in labels])
        pred_proteins_without_lbs = np.array([max(label) == 0 for label in y_pred])
        both_without_lbs = true_proteins_without_lbs & pred_proteins_without_lbs

        self.results["True proteins without LBS"] = true_proteins_without_lbs.sum()
        self.results["Pred proteins without LBS"] = pred_proteins_without_lbs.sum()
        self.results["Pred&True proteins without LBS"] = (true_proteins_without_lbs & pred_proteins_without_lbs).sum()

        accuracies = [accuracy_score(label, pred) for label, pred in zip(labels, y_pred) if label.shape[0]]
        confidence_interval = 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))

        self.results["Session Accuracy Median"] = np.median(accuracies)
        self.results["Session Accuracy Mean"] = np.mean(accuracies)
        self.results["Session Accuracy CI min"] = np.mean(accuracies) - confidence_interval
        self.results["Session Accuracy CI max"] = np.mean(accuracies) + confidence_interval

        precisions = [precision_score(label, pred, zero_division=int(both_without_lbs[i]))
                      for label, pred, i in zip(labels, y_pred, range(len(labels))) if label.shape[0]]
        confidence_interval = 1.96 * np.std(precisions) / np.sqrt(len(precisions))
        self.results["Session Precision Median"] = np.median(precisions)
        self.results["Session Precision Mean"] = np.mean(precisions)
        self.results["Session Precision CI min"] = np.mean(precisions) - confidence_interval
        self.results["Session Precision CI max"] = np.mean(precisions) + confidence_interval

        recalls = [recall_score(label, pred, zero_division=int(both_without_lbs[i]))
                   for label, pred, i in zip(labels, y_pred, range(len(labels))) if label.shape[0]]
        confidence_interval = 1.96 * np.std(recalls) / np.sqrt(len(recalls))
        self.results["Session Recall Median"] = np.median(recalls)
        self.results["Session Recall Mean"] = np.mean(recalls)
        self.results["Session Recall CI min"] = np.mean(recalls) - confidence_interval
        self.results["Session Recall CI max"] = np.mean(recalls) + confidence_interval

        f1_scores = [f1_score(label, pred, zero_division=int(both_without_lbs[i]))
                     for label, pred, i in zip(labels, y_pred, range(len(labels))) if label.shape[0]]
        confidence_interval = 1.96 * np.std(f1_scores) / np.sqrt(len(f1_scores))
        self.results["Session F1_score Median"] = np.median(f1_scores)
        self.results["Session F1_score Mean"] = np.mean(f1_scores)
        self.results["Session F1_score CI min"] = np.mean(f1_scores) - confidence_interval
        self.results["Session F1_score CI max"] = np.mean(f1_scores) + confidence_interval

        return self

    def save_to_file(self, file_name: Path):
        print(self.results)
        tabulated = tabulate(self.results.items(), tablefmt="github", headers=["Metric", "Value"])
        with open(file_name, "w") as file:
            file.write(tabulated)
        print(tabulated)
        json_file = file_name.with_suffix(".json")
        roc_curve_results = copy.deepcopy(self.results)
        roc_curve_results.update(
            {"fpr": self.fpr.tolist(), "tpr": self.tpr.tolist(), "thresholds": self.thresholds.tolist()})
        try:
            with open(json_file, "w") as file:
                json.dump(roc_curve_results, file, indent=4)
        except TypeError:
            with open(json_file, "w") as file:
                roc_curve_results = {str(k): str(v) for k, v in roc_curve_results.items()}
                json.dump(roc_curve_results, file, indent=4)
        plt.plot(self.fpr, self.tpr)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.5, 0.2, f"AUC: {self.results['AUC']:.4f}", fontsize=14,
                 verticalalignment='top', bbox=props)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC curve - {self.model.name}")
        plt.savefig(file_name.with_suffix(".png"))
        plt.clf()
        return self

    def print(self):
        print(tabulate(self.results.items(), tablefmt="github", headers=["Metric", "Value"]))
        roc_curve_results = copy.deepcopy(self.results)
        roc_curve_results.update(
            {"fpr": self.fpr.tolist(), "tpr": self.tpr.tolist(), "thresholds": self.thresholds.tolist()})
        print(roc_curve_results)
        return self

    def _get_basic_data(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        # TODO: Implement a data loader
        with open(self.config.test_extracted, "rb") as file:
            arffs = pickle.load(file)
        print("Data loaded.")
        print("Preparing data...")
        labels = list(map(lambda x: np.vectorize(booleanize)(x["@@class@@"].to_numpy()), arffs))
        data = list(map(lambda x: x.drop("@@class@@", axis=1), arffs))
        if self.config.test_size:
            data = data[:self.config.test_size]
            labels = labels[:self.config.test_size]
        return data, labels

    def _get_surroundings_data(self) -> Tuple[np.array, np.array]:
        data: np.array
        labels: np.array
        with open(self.config.test_surroundings, "rb") as file:
            data, labels = pickle.load(file)
        if self.config.test_size:
            points_length = sum(self.config.train_lengths[:self.config.train_size])
            data = data[:points_length]
            labels = labels[:points_length]
        labels = np.vectorize(booleanize)(labels)
        return data, labels


def booleanize(x: np.array):
    if x in (True, 1, b'1', b'True'):
        return True
    if x in (False, 0, b'0', b'False'):
        return False
    raise ValueError
