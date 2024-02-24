import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.common.GenericProteinModel import GenericProteinModel


class RandomForestModel(GenericProteinModel):
    @property
    def name(self) -> str:
        return "RFC_baseline"

    def __init__(self, random_forest: RandomForestClassifier, cutoff: float):
        self.cutoff: float = cutoff
        self.random_forest: RandomForestClassifier = random_forest

    def predict(self, protein: np.ndarray) -> np.ndarray:
        return self.random_forest.predict_proba(protein)[:, 1] > self.cutoff
