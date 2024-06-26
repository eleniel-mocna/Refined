"""
This is a python script that trains a NN model. It accepts a CLI argument:
--tune-hyperparameters: If specified, hyperparameter tuning will be done,
    otherwise previously found hyperparameters will be used.
"""
import sys
from typing import Any, Optional, Dict

import keras
import keras_tuner as kt
import numpy as np
from keras_tuner import HyperParameters

from config.config import Config
from models.common.ProteinModel import SurroundingsProteinModel
from models.common.run_script_functions import train_surroundings_model

np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer


class BigNN(SurroundingsProteinModel):
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def predict_surroundings(self, protein: np.ndarray) -> np.ndarray:
        return (self.model.predict(protein) > 0.5).flatten()

    def predict_surroundings_proba(self, protein: np.ndarray) -> np.ndarray:
        return self.model.predict(protein).flatten()

    @property
    def name(self) -> str:
        return "surroundings_NN"


def nn_model_builder(hp: HyperParameters):
    hp_initial_size = hp.Int("initial_size", min_value=16, max_value=4096, sampling="log", step=2)
    hp_growth_factor = hp.Float("growth_factor", min_value=0.25, max_value=1.5, step=0.05)
    hp_layers = hp.Int("layers", min_value=1, max_value=4)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5])

    model = tf.keras.models.Sequential([])
    model.add(InputLayer(input_shape=(38 * Config.default().surroundings_size)))
    for i in range(hp_layers):
        try:
            model.add(Dense(units=int(hp_initial_size * (hp_growth_factor ** i)), activation=relu, name=f"Dense_{i}"))
        except ValueError:
            pass
    model.add(Dense(units=1, name='logits', activation="sigmoid"))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
    return model


def generate_nn_model(data,
                      labels,
                      hyperparams: Optional[Dict[str, Any]] = None) -> BigNN:
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    if hyperparams is not None:
        hyper_parameters = kt.HyperParameters()
        for key, value in hyperparams.items():
            hyper_parameters.Fixed(key, value)
        model = nn_model_builder(hyper_parameters)
        model.fit(data, labels, epochs=50, validation_split=0.2, callbacks=[stop_early])
        return BigNN(model)

    tuner = kt.Hyperband(nn_model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         )
    tuner.search(data, labels, epochs=50, validation_split=0.2, callbacks=[stop_early])
    print(f"Best hyperparams: \n{tuner.get_best_hyperparameters()[0].values}")
    model = tuner.get_best_models(1)[0]
    return BigNN(model)


best_hyperparams = {'initial_size': 32, 'growth_factor': 0.3, 'layers': 1, 'learning_rate': 0.0005}


def main(tune_hyperparameters: bool = False):
    train_surroundings_model(generate_nn_model,
                             hyperparams=None if tune_hyperparameters else best_hyperparams)


if __name__ == '__main__':
    main(tune_hyperparameters="--tune-hyperparameters" in sys.argv)
