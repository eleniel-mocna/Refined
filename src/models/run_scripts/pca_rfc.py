import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from config.config import Config
from models.common.ProteinModel import SurroundingsProteinModel
from models.common.cutoff import get_best_cutoff
from models.evaluation.ModelEvaluator import ModelEvaluator, booleanize

np.set_printoptions(threshold=sys.maxsize)
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV


class PcaRfc(SurroundingsProteinModel):
    def predict_surroundings(self, protein: np.ndarray) -> np.ndarray:
        return self.rfc.predict_proba(protein)[:, 1] > self.cutoff

    def predict_surroundings_proba(self, protein: np.ndarray) -> np.ndarray:
        return self.rfc.predict_proba(protein)

    @property
    def name(self) -> str:
        return "pca_rfc" + self.name_suffix

    def __init__(self, rfc: RandomForestClassifier, pca: PCA, cutoff: float, name_suffix:str = ""):
        self.name_suffix = name_suffix
        self.cutoff: float = cutoff
        self.rfc: RandomForestClassifier = rfc
        self.pca: PCA = pca

    @staticmethod
    def from_data(data: np.ndarray, labels: np.ndarray, pca_dimension: int) -> 'PcaRfc':
        """
        Create a GeneralModel implementation using a RandomForestClassifier.
        @param pca_dimension: Number of dimensions to reduce to
        @param data: Numpy array of atom neighborhoods
        @param labels: Numpy array of labels for each atom
        @return: trained RandomForestModel
        """
        train_data, test_data, train_labels, test_labels = \
            train_test_split(data, labels, test_size=0.20, random_state=42)
        pca = PCA(pca_dimension)
        reduced_train_data = pca.fit_transform(train_data)
        reduced_test_data = pca.transform(test_data)

        print("Data prepared, training RFC...")
        best_params = PcaRfc.get_best_hyperparameters(reduced_train_data, train_labels)
        print(f"Best params: {best_params}")
        random_forest = RandomForestClassifier(n_jobs=15,
                                                verbose=0,
                                                **best_params)
        random_forest.fit(reduced_train_data, train_labels)
        print("RFC trained, finding best cutoff...")
        best_cutoff = get_best_cutoff(reduced_test_data, test_labels, random_forest)
        print(best_cutoff)
        return PcaRfc(random_forest, pca, best_cutoff, f"_{pca_dimension}")

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

    for pca_size in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        print(f"Training PCA with size {pca_size}")
        rfc_surrounding_model = PcaRfc.from_data(np.array(data), np.array(labels), pca_size)
        rfc_surrounding_model.save_model()
        (ModelEvaluator(rfc_surrounding_model)
         .calculate_basic_metrics()
         .calculate_session_metrics()
         .save_to_file(rfc_surrounding_model.get_result_folder() / "metrics.txt"))


if __name__ == '__main__':
    main()
