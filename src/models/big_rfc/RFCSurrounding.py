import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dataset.surroundings_calculation.surroundings_extractor import SurroundingsExtractor
from models.common.GenericProteinModel import GenericProteinModel


class RFCSurrounding(GenericProteinModel):
    @property
    def name(self) -> str:
        return "rfc_surrounding"

    def __init__(self, rfc: RandomForestClassifier, cutoff: float):
        self.cutoff: float = cutoff
        self.rfc: RandomForestClassifier = rfc

    def predict(self, protein: pd.DataFrame) -> np.ndarray:
        input_dataset = SurroundingsExtractor.get_complete_dataset(protein, 30)
        input_data = [sample.drop("@@class@@", axis=1, errors="ignore").to_numpy().flatten() for sample in input_dataset]
        return self.rfc.predict_proba(input_data)[:, 1] > self.cutoff

    def predict_proba(self, protein: pd.DataFrame) -> np.ndarray:
        input_dataset = SurroundingsExtractor.get_complete_dataset(protein, 30)
        input_data = [sample.drop("@@class@@", axis=1).to_numpy().flatten() for sample in input_dataset]
        return self.rfc.predict_proba(input_data)[:, 1]
