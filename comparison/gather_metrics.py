from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from comparison.GenericProteinModel import GenericProteinModel


def gather_metrics(model: GenericProteinModel,
                   data: Iterable[np.array],
                   labels: Iterable[np.array],
                   output_file: str):
    """
    Gather metrics for a model on a set of proteins
    @param model: model to evaluate
    @param data: list of proteins
    @param labels: list of labels for each atom in the protein
    @param output_file: file to write metrics to
    """
    metrics_gatherer = MetricsGatherer()
    data_list = list(data)
    labels_list = list(labels)
    y_pred = model.predict(data_list)
    for i in range(len(data_list)):
        metrics_gatherer.add(y_pred[i], labels_list[i])
    metrics_gatherer.write(output_file)
    return metrics_gatherer


class MetricsGatherer:
    def __init__(self):
        self.accuracies = []
        self.f1s = []
        self.precisions = []
        self.recalls = []
        self.roc_aucs = []

    def add(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.accuracies.append(accuracy_score(y_true, y_pred))
        self.f1s.append(f1_score(y_true, y_pred))
        self.precisions.append(precision_score(y_true, y_pred))
        self.recalls.append(recall_score(y_true, y_pred))
        self.roc_aucs.append(roc_auc_score(y_true, y_pred))

    def write(self, output_file: str) -> None:
        with open(output_file, "w") as file:
            file.write(f"Mean accuracy: {np.mean(self.accuracies)}, var: {np.var(self.accuracies)}\n")
            file.write(f"Mean f1: {np.mean(self.f1s)}, var: {np.var(self.f1s)}\n")
            file.write(f"Mean precision: {np.mean(self.precisions)}, var: {np.var(self.precisions)}\n")
            file.write(f"Mean recall: {np.mean(self.recalls)}, var: {np.var(self.recalls)}\n")
            file.write(f"Mean roc_auc: {np.mean(self.roc_aucs)}, var: {np.var(self.roc_aucs)}\n")
