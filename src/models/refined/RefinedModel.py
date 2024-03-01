import numpy as np
import pandas as pd
import tensorflow as tf

from dataset.surroundings_calculation.surroundings_extractor import SurroundingsExtractor
from models.common.ProteinModel import ProteinModel
from models.refined.Refined import Refined


class RefinedModel(ProteinModel):
    @property
    def name(self) -> str:
        return "Refined"

    def __init__(self, model: tf.keras.Model, refined: Refined):
        self.refined = refined
        self.model = model

    def predict(self, protein: pd.DataFrame) -> np.ndarray:
        input_dataset = SurroundingsExtractor.get_complete_dataset(protein, 30)

        input_data = pd.concat(map(lambda x: x.drop("@@class@@", axis=1).numpy().flatten(), input_dataset))
        return self.model.predict(self.refined.transform(input_data))

    def save(self, path):
        self.model.save_weights(path + "/weights.h5")
        self.model.save(path + "/model.json")
        self.model = None

    def load(self, path):
        if self.model is None:
            self.model = tf.keras.models.load_model(path + "/model.json")
            self.model.load_weights(path + "/weights.h5")
        else:
            print("Model already loaded")
