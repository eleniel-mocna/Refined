import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.common.ProteinModel import SurroundingsProteinModel
from models.refined.image_transformer import ImageTransformer


class RefinedModel(SurroundingsProteinModel):
    def predict_surroundings(self, protein: np.ndarray) -> np.ndarray:
        return self.model.predict(self.refined.transform(protein)).flatten() > 0.5

    def predict_surroundings_proba(self, protein: np.ndarray) -> np.ndarray:
        return self.model.predict(self.refined.transform(protein)).flatten()

    @property
    def name(self) -> str:
        return "Refined" + self.name_suffix

    def __init__(self, model: tf.keras.Model, refined: ImageTransformer, name: str = ""):
        self.refined = refined
        self.model = model
        self.name_suffix = name

    def save_predictor(self, path):
        self.model.save_weights(path / "weights.h5")
        self.model.save(path / "model.json")
        self.model = None

    def load_predictor(self, path):
        if self.model is None:
            self.model = tf.keras.models.load_model(path / "model.json")
            self.model.load_weights(path / "weights.h5")
        else:
            print("Model already loaded")

    def save_model(self):
        folder = self.get_result_folder()
        self.save_predictor(folder)
        super().save_model()

    @staticmethod
    def from_folder(folder: Path):
        refined_model:RefinedModel = pickle.load(open(folder/"model.pkl", "rb"))
        refined_model.load_predictor(folder)
        return refined_model