import os
import sys

import numpy as np

from refined.RefinedModel import RefinedModel

np.set_printoptions(threshold=sys.maxsize)
import pickle
from refined.Refined import Refined
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import MaxPool2D


def create_model():
    model = tf.keras.models.Sequential([])
    model.add(InputLayer(input_shape=(38, 30, 1)))
    model.add(Conv2D(filters=5, kernel_size=(5, 5), activation=relu))
    model.add(MaxPool2D(strides=2, pool_size=(2, 2)))
    # model.add(Conv2D(filters=64, kernel_size=5, activation=relu))
    # model.add(MaxPool2D(strides=2, pool_size=(2, 2)))
    model.add(Flatten(name='Flatten'))
    # model.add(Dense(units=32, activation=relu))
    model.add(Dense(units=1, name='logits', activation="sigmoid"))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def generate_refined_model_from_dataset():
    with open("refined/data.pckl", "rb") as file:
        data = pickle.load(file)

    with open("refined/labels.pckl", "rb") as file:
        labels = pickle.load(file)
    return generate_refined_model(data, labels)


def generate_refined_model(data, labels):
    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, labels, test_size=0.20, random_state=42)
    refined = Refined(train_data, 38, 30, "temp", hca_starts=6)
    refined.run()
    refined_train_data = refined.transform(train_data)
    refined_test_data = refined.transform(test_data)
    model = create_model()
    history = model.fit(refined_train_data, train_labels, validation_data=(refined_test_data, test_labels), epochs=10)
    return RefinedModel(model, refined), history


if __name__ == '__main__':
    print("Loading data...")
    folder = "refined/NN"
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_file = os.path.join(folder, "metrics.txt")
    with open("surroundings_dataset_1700.pckl", "rb") as file:
        data, labels = pickle.load(file)
    data = np.array(data)
    labels = np.array(labels)
    refined_model, history = generate_refined_model(data, labels)
    refined_model.save(folder)
    with open(os.path.join(folder, "refinedModel.pckl"), "wb") as file:
        pickle.dump(refined_model, file)
