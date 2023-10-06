import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from comparison.GenericProteinModel import GenericProteinModel
from utils.extract_surroundings import get_complete_dataset


class RFCSurrounding(GenericProteinModel):
    def __init__(self, rfc: RandomForestClassifier, cutoff: float):
        self.cutoff: float = cutoff
        self.rfc: RandomForestClassifier = rfc

    def predict(self, protein: pd.DataFrame) -> np.ndarray:
        input_dataset = get_complete_dataset(protein, 30)
        input_data = [sample.drop("@@class@@", axis=1).to_numpy().flatten() for sample in input_dataset]
        return self.rfc.predict_proba(input_data)[:, 1] > self.cutoff

    def predict_proba(self, protein: pd.DataFrame) -> np.ndarray:
        input_dataset = get_complete_dataset(protein, 30)
        input_data = [sample.drop("@@class@@", axis=1).to_numpy().flatten() for sample in input_dataset]
        return self.rfc.predict_proba(input_data)[:, 1]
