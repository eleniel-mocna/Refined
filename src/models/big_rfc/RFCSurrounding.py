import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dataset.surroundings_calculation.surroundings_extractor import SurroundingsExtractor
from models.cutoff import get_best_cutoff
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

    @staticmethod
    def from_data(data: np.ndarray, labels: np.ndarray):
        """
        Create a GeneralModel implementation using a RandomForestClassifier.
        @param data: Numpy array of atom neighborhoods
        @param labels: Numpy array of labels for each atom
        @return: trained RandomForestModel
        """
        train_data, test_data, train_labels, test_labels = \
            train_test_split(data, labels, test_size=0.20, random_state=42)

        train_data = train_data
        test_data = test_data
        print("Data prepared, training RFC...")
        random_forest: RandomForestClassifier = RandomForestClassifier(max_depth=5, n_jobs=15, verbose=3,
                                                                       n_estimators=15)
        random_forest.fit(train_data, train_labels)
        print("RFC trained, finding best cutoff...")
        best_cutoff = get_best_cutoff(test_data, test_labels, random_forest)
        print(best_cutoff)
        return RFCSurrounding(random_forest, best_cutoff)