import numpy as np
import pandas as pd
import tensorflow as tf

from refined.Refined import Refined
from comparison.GenericProteinModel import GenericProteinModel
from utils.extract_surroundings import get_complete_dataset


class RefinedModel(GenericProteinModel):
    def __init__(self, model: tf.keras.Model, refined: Refined, ):
        self.refined = refined
        self.model = model

    def predict(self, protein: pd.DataFrame) -> np.ndarray:
        input_dataset = get_complete_dataset(protein, 30)

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
