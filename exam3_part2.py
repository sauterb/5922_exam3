# Bailey Sauter
# CSCI 5922 Fall 2023
# Exam 3 Part 2

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

if __name__ == '__main__':
    # SAUTER EXAM 3 PART 2.1
    # Build sequential ANN model
    NN_Model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(4),
        tf.keras.layers.Dense(4, activation='sigmoid'), # Dense layer
        tf.keras.layers.Dense(3, activation='relu'), # Dense layer 2
        tf.keras.layers.Dense(3, activation='softmax'), # Output Layer
    ])

    NN_Model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    NN_Model.summary()

    # SAUTER EXAM 3 PART 2.2
    # Build Sequential RNN model
    RNN_Model = tf.keras.models.Sequential([
        keras.layers.Conv2D(filters=2, kernel_size=(3, 3), activation='relu', input_shape=(30,30,1), padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    RNN_Model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

    RNN_Model.summary()

