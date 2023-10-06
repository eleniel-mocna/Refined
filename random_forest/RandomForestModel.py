import numpy as np
from sklearn.ensemble import RandomForestClassifier

from comparison.GenericProteinModel import GenericProteinModel


class RandomForestModel(GenericProteinModel):
    def __init__(self, random_forest: RandomForestClassifier, cutoff: float):
        self.cutoff: float = cutoff
        self.random_forest: RandomForestClassifier = random_forest

    def predict(self, protein: np.ndarray) -> np.ndarray:
        return self.random_forest.predict_proba(protein)[:, 1] > self.cutoff
