# Bailey Sauter
# CSCI 5922 Fall 2023
# Exam 3 Part 2

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function for plotting loss and accuracy across epochs
def twoAxisEpochsPlot(accuracy, loss, num_epochs):
    # Create the first axis (left y-axis)
    fig, ax1 = plt.subplots()
    x = np.linspace(1,num_epochs,num_epochs)

    # Plot the accuracy on the left y-axis
    ax1.plot(x, accuracy, color='blue', label='Model Accuracy')
    ax1.set_xlabel('Epoch #')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params('y', colors='blue')

    # Create the second axis (right y-axis)
    ax2 = ax1.twinx()

    # Plot the loss on the right y-axis
    ax2.plot(x, loss, color='red', label='Loss Function')
    ax2.set_ylabel('Loss', color='red')
    ax2.tick_params('y', colors='red')

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Model Accuracy and Loss over Epochs')
    plt.show()

# Function for creating good-looking confusion matrices
def make_confusion_matrix(y_hat, y, model_name, labels):
    # Convert one-hot encodings into integer encodings
    max_values = y.argmax(axis=1)
    max_values_hat = y_hat.argmax(axis=1)
    cm = confusion_matrix(max_values_hat, max_values)

    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
    ax.set_xlabel('True labels')
    ax.set_ylabel('Predicted labels')
    ax.set_title(f'{model_name} model Confusion Matrix'
                 f'\n0 = {labels[0]}'
                 f'\n1 = {labels[1]}'
                 f'\n2 = {labels[2]}')
    plt.show()

if __name__ == '__main__':
    # Boolean variables for which models to run
    ann = True
    cnn = True
    lstm = True

    # Import the data
    filename = "Final_News_DF_Labeled_ExamDataset.csv"
    df = pd.read_csv(filename)

    # Extract labels and word columns
    labels = df["LABEL"].values
    word_columns = df.drop("LABEL", axis=1).values

    # One hot encode labels
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(labels)
    labels_plainspeak = label_binarizer.classes_

    # Convert word columns to np array
    X = np.array(word_columns)

    # ---ANN---
    if ann:
        # split data up
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

        print("---ANN MODEL RESULTS---")
        NN_Model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(300),
            tf.keras.layers.Dense(48, activation='relu'), # Dense layer
            tf.keras.layers.Dense(3, activation='softmax'), # Output Layer
        ])

        # Compile model
        NN_Model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        NN_Model.summary()

        # train ANN
        print("-STARTING TRAINING-")
        training = NN_Model.fit(X_train, y_train, batch_size=128, epochs=32, use_multiprocessing=True, verbose=True)
        accuracy = training.history['accuracy']
        loss = training.history['loss']

        # Plot training process
        twoAxisEpochsPlot(accuracy, loss, len(loss))

        # Test model and make confusion matrix
        print("-STARTING TESTING-")
        predictions = NN_Model.predict(X_test, batch_size=32, use_multiprocessing=True, verbose=True)
        evaluation = NN_Model.evaluate(X_test, y_test, batch_size=32, use_multiprocessing=True, verbose=True)
        make_confusion_matrix(predictions, y_test, model_name="ANN", labels=labels_plainspeak)

    # ---CNN---
    if cnn:
        # Use pre-trained embedding function to embed words - should improve performance in sequential models
        all_words = df.columns[1:].values

        # Make all the samples into lists of the words present within them
        X_tokenized = []
        for text in X:
            X_tokenized.append([])
            current_x_index = len(X_tokenized)-1
            for i in range(len(text)):
                if text[i]:
                    for j in range(text[i]):
                        X_tokenized[current_x_index].append(all_words[i])

        # Train Word2Vec model to embed words as 2D vectors
        vector_size = 2
        word2vec_model = Word2Vec(sentences=X_tokenized, vector_size=vector_size, window=5, min_count=1, workers=4)

        # Minmax normalize all word vectors and multiply by 99 so they can be placed on a (100,100) array
        all_words_vectorized = np.asarray([word2vec_model.wv[word] for word in all_words])
        all_words_vectorized = 99 * MinMaxScaler().fit_transform(all_words_vectorized)

        # Iterate over samples and add count for each word in the sample at each coordinate on a 2D plane
        X_embedded = np.empty((len(X), 100, 100))
        for i in range(len(X)):
            for j in range(len(all_words_vectorized)):
                x_coordinate = int(all_words_vectorized[j][0])
                y_coordinate = int(all_words_vectorized[j][1])
                X_embedded[i][x_coordinate][y_coordinate] = X[i][j]
        # Now we have a 1 channel, 100x100 array for each sample, ideal for CNN
        X_embedded = X_embedded[:, :, :, np.newaxis]

        # Split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.2, random_state=99)

        print("---CNN MODEL RESULTS---")
        NN_Model = tf.keras.models.Sequential([
            # Run 2D CNN on our 100x100 arrays for each sample text
            keras.layers.Conv2D(filters=8, kernel_size=(3, 3),strides=(1,1), activation='relu', input_shape=X_train[0].shape, padding='same'),
            keras.layers.MaxPool2D(pool_size=(2,2), padding='same'),
            keras.layers.Flatten(),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        NN_Model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

        NN_Model.summary()

        # train
        training = NN_Model.fit(X_train, y_train, batch_size=128, epochs=12, use_multiprocessing=True, verbose=True)
        accuracy = training.history['accuracy']
        loss = training.history['loss']
        twoAxisEpochsPlot(accuracy, loss, len(loss))

        # test + generate confusion matrix
        predictions = NN_Model.predict(X_test, batch_size=32, use_multiprocessing=True, verbose=True)
        evaluation = NN_Model.evaluate(X_test, y_test, batch_size=32, use_multiprocessing=True, verbose=True)
        make_confusion_matrix(predictions, y_test, "CNN", labels=labels_plainspeak)

    if lstm:
        all_words = df.columns[1:].values

        X_tokenized = []
        for text in X:
            X_tokenized.append([])
            current_x_index = len(X_tokenized) - 1
            for i in range(len(text)):
                if text[i]:
                    for j in range(text[i]):
                        X_tokenized[current_x_index].append(all_words[i])
        # Train Word2Vec model, this time use 30 features for each word
        vector_size = 30
        word2vec_model = Word2Vec(sentences=X_tokenized, vector_size=vector_size, window=20, min_count=1, workers=4)

        # Because there is no sequential data for an RNN to utilize
        # Do our best to create a sequence by ordering words from least to most used in that text
        for i in range(len(X)):
            X[i] = np.argsort(X[i])

        # Convert "sequences" of words to Word2Vec embeddings
        X_embedded = np.empty((len(X), len(X[0]), vector_size))
        for i in range(len(X)):
            for j in range(len(X[i])):
                foo = all_words[X[i][j]]
                X_embedded[i][j] = word2vec_model.wv[all_words[X[i][j]]]

        # Split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.2, random_state=99)

        # ---LSTM---
        print("---LSTM MODEL RESULTS---")
        NN_Model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X_train[0].shape)),
            tf.keras.layers.LSTM(units=50),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        NN_Model.build()

        NN_Model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

        NN_Model.summary()

        # train
        training = NN_Model.fit(X_train, y_train, batch_size=128, epochs=50, use_multiprocessing=True, verbose=True)
        accuracy = training.history['accuracy']
        loss = training.history['loss']
        twoAxisEpochsPlot(accuracy, loss, len(loss))

        # test + confusion matrix
        predictions = NN_Model.predict(X_test, batch_size=32, use_multiprocessing=True, verbose=True)
        evaluation = NN_Model.evaluate(X_test, y_test, batch_size=32, use_multiprocessing=True, verbose=True)
        make_confusion_matrix(predictions, y_test, "LSTM", labels=labels_plainspeak)