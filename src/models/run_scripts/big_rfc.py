import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config.config import Config
from models.common.ProteinModel import SurroundingsProteinModel
from models.evaluation.ModelEvaluator import ModelEvaluator, booleanize

np.set_printoptions(threshold=sys.maxsize)
import pickle
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
    def from_data(data: np.ndarray, labels: np.ndarray) -> 'RFCSurrounding':
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
        random_forest: RandomForestClassifier = RandomForestClassifier(n_jobs=15, verbose=3,
                                                                       n_estimators=500,
                                                                       min_samples_split=5,
                                                                       max_features='sqrt',
                                                                       max_depth=10)

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
        return tuner.best_params_


def main():
    config = Config.get_instance()
    with open(config.train_surroundings, "rb") as file:
        data, labels = pickle.load(file)
    labels = np.vectorize(booleanize)(labels)

    rfc_surrounding_model = RFCSurrounding.from_data(np.array(data), np.array(labels))
    rfc_surrounding_model.save_model()
    (ModelEvaluator(rfc_surrounding_model)
     .calculate_basic_metrics()
     .calculate_session_metrics()
     .save_to_file(rfc_surrounding_model.get_result_folder() / "metrics.txt")
     .print())


if __name__ == '__main__':
    main()
