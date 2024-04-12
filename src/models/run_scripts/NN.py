import sys
from typing import Any

import numpy as np

from config.config import Config
from models.common.ProteinModel import SurroundingsProteinModel, ProteinModel
from models.evaluation.ModelEvaluator import ModelEvaluator
from models.refined.RefinedModel import RefinedModel

np.set_printoptions(threshold=sys.maxsize)
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer

class BigNN(SurroundingsProteinModel):
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def predict_surroundings(self, protein: np.ndarray) -> np.ndarray:
        return self.model.predict(protein) > 0.5

    def predict_surroundings_proba(self, protein: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(protein)

    @property
    def name(self) -> str:
        return "surroundings_NN"


def create_model():
    model = tf.keras.models.Sequential([])
    model.add(InputLayer(input_shape=(38 * 30)))
    model.add(Dense(units=30, activation=relu))
    model.add(Dense(units=1, name='logits', activation="sigmoid"))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def generate_refined_model_from_dataset():
    with open("refined/data.pckl", "rb") as file:
        data = pickle.load(file)

    with open("refined/labels.pckl", "rb") as file:
        labels = pickle.load(file)
    return train_model(data, labels)


def train_model(data, labels) -> tuple[ProteinModel, Any]:
    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels, test_size=0.20, random_state=42)
    model = create_model()
    history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=1)
    return BigNN(model), history


def main():
    with open(config.train_surroundings, "rb") as file:
        data, labels = pickle.load(file)

    rfc_surrounding_model, _ = train_model(np.array(data), np.array(labels))
    rfc_surrounding_model.save_model()
    (ModelEvaluator(rfc_surrounding_model)
     .calculate_basic_metrics()
     .calculate_session_metrics()
     .save_to_file(rfc_surrounding_model.get_result_folder() / "metrics.txt"))


config = Config.get_instance()

if __name__ == '__main__':
    main()
