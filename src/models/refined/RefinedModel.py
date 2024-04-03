import pickle
from pathlib import Path
from typing import Any

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.activations import relu
from keras.layers import Conv2D, Flatten, Dense, InputLayer
from keras_tuner import HyperParameters
from keras_tuner.src.backend import keras

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

    def load_predictor(self, path):
        if self.model is None:
            self.model = tf.keras.models.load_model(path / "model.json")
            self.model.load_weights(path / "weights.h5")
        else:
            print("Model already loaded")

    def save_model(self):
        folder = self.get_result_folder()
        self.save_predictor(folder)
        predictor = self.model
        self.model = None
        super().save_model()
        self.model = predictor


    @staticmethod
    def from_folder(folder: Path):
        refined_model: RefinedModel = pickle.load(open(folder / "model.pkl", "rb"))
        refined_model.load_predictor(folder)
        return refined_model


def generate_refined_model(data, labels, image_transformer: ImageTransformer) -> tuple[RefinedModel, Any]:
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    tuner = kt.Hyperband(cnn_model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         )
    tuner.search(image_transformer.transform(data), labels, epochs=50, validation_split=0.2, callbacks=[stop_early])
    print(f"Best hyperparams: \n{tuner.get_best_hyperparameters()[0].values}")
    model = tuner.get_best_models(1)[0]
    return RefinedModel(model, image_transformer), tuner

def cnn_model_builder(hp: HyperParameters):
    hp_initial_size = hp.Int("initial_size", min_value=4, max_value=128, sampling="log", step=2)
    hp_growth_factor = hp.Float("growth_factor", min_value=0.5, max_value=2, step=0.1)
    hp_last_dense = hp.Int("last_dense", min_value=1, max_value=128, sampling="log", step=2)
    hp_cnn_layers = hp.Int("cnn_layers", min_value=1, max_value=4)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5])

    model = tf.keras.models.Sequential([])
    model.add(InputLayer(input_shape=(38, 30, 1)))
    for i in range(hp_cnn_layers):
        model.add(
            Conv2D(filters=hp_initial_size * (hp_growth_factor ** i), kernel_size=(3, 3), strides=2, activation=relu,
                   name=f"CNN_{i}"))
    model.add(Flatten(name='Flatten'))
    if hp_last_dense > 1:
        model.add(Dense(units=hp_last_dense, activation=relu, name="Dense"))
    model.add(Dense(units=1, name='logits', activation="sigmoid"))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
    return model
