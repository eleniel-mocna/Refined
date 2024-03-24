import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dataset.surroundings_calculation.surroundings_extractor import SurroundingsExtractor
from models.common.ProteinModel import ProteinModel, SurroundingsProteinModel


class RFCSurrounding(SurroundingsProteinModel):
    def predict_surroundings(self, protein: np.ndarray) -> np.ndarray:
        return self.rfc.predict_proba(protein)[:, 1] > self.cutoff

    def predict_surroundings_proba(self, protein: np.ndarray) -> np.ndarray:
        return self.rfc.predict_proba(protein)

    @property
    def name(self) -> str:
        return "rfc_surrounding"

    def __init__(self, rfc: RandomForestClassifier, cutoff: float):
        self.cutoff: float = cutoff
        self.rfc: RandomForestClassifier = rfc
