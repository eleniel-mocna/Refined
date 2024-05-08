"""
This is a python script that trains a surroundings RFC model. It accepts a CLI argument:
--tune-hyperparameters: If specified, hyperparameter tuning will be done,
    otherwise previously found hyperparameters will be used.
"""
import sys
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.common.ProteinModel import SurroundingsProteinModel
from models.common.run_script_functions import train_surroundings_model

np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split, GridSearchCV


class RFCSurrounding(SurroundingsProteinModel):
    def predict_surroundings(self, protein: np.ndarray) -> np.ndarray:
        return self.rfc.predict_proba(protein)[:, 1] > 0.5

    def predict_surroundings_proba(self, protein: np.ndarray) -> np.ndarray:
        return self.rfc.predict_proba(protein)[:, 1]

    @property
    def name(self) -> str:
        return "rfc_surrounding"

    def __init__(self, rfc: RandomForestClassifier):
        self.rfc: RandomForestClassifier = rfc

    @staticmethod
    def RFCSurrounding_from_data(data: np.ndarray, labels: np.ndarray,
                                 hyperparameters: Optional[Dict] = None) -> 'RFCSurrounding':
        """
        Create a GeneralModel implementation using a RandomForestClassifier.
        @param hyperparameters: If None, does hyperparameter tuning
        @param data: Numpy array of atom neighborhoods
        @param labels: Numpy array of labels for each atom
        @return: trained RandomForestModel
        """
        train_data, test_data, train_labels, test_labels = \
            train_test_split(data, labels, test_size=0.20, random_state=42)

        print("Data prepared, training RFC...")
        random_forest: RandomForestClassifier = (
            RandomForestClassifier(n_jobs=15, verbose=3, **hyperparameters) if hyperparameters
            else RFCSurrounding.get_best_hyperparameters(data, labels))

        random_forest.fit(train_data, train_labels)
        return RFCSurrounding(random_forest)

    @staticmethod
    def get_best_hyperparameters(data: np.ndarray, labels: np.ndarray):
        param_grid = {'max_depth': [3, 5, 10, None],
                      'min_samples_split': [2, 5, 10],
                      'n_estimators': [100, 200, 300, 500, 1000],
                      "max_features": ["sqrt", "log2", None]}
        estimator = RandomForestClassifier(n_jobs=15,
                                           verbose=0,
                                           max_features=6)
        tuner = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)
        tuner.fit(data, labels)
        return tuner.best_estimator_


def main(tune_hyperparameters: bool = False):
    train_surroundings_model(RFCSurrounding.RFCSurrounding_from_data,
                             hyperparameters=None if tune_hyperparameters else {"n_estimators": 500,
                                                                                "min_samples_split": 5,
                                                                                "max_features": 'sqrt',
                                                                                "max_depth": 10})


if __name__ == '__main__':
    main(tune_hyperparameters="--tune-hyperparameters" in sys.argv)
