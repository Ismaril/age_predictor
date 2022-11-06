import os
import numpy as np
import pandas as pd
import constants as c
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import utils
from sklearn.utils.class_weight import compute_class_weight


class ModelTraining:
    def __init__(self):
        self.features = None
        self.labels = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_valid = None
        self.y_valid = None
        self.model = None
        self.batch_size = None
        self.epochs = None
        self.history = None

    def load_data(self, path_features: str, path_labels: str):
        """
        Load features and labels.

        :return: None
        """
        assert path_features.endswith(".npy"), "File must be *.npy"
        assert path_labels.endswith(".csv"), "File must be *.csv"

        features = np.load(path_features)
        labels = pd.read_table(path_labels, delimiter=",")
        labels = np.array(labels["labels"])

        self.features = features
        self.labels = labels

    def change_dtypes(self):
        """
        Change data types to float.

        :return: None
        """

        self.features = self.features.astype(np.float16)

    def one_hot_encode_labels(self):
        """
        Set number of columns based on number of classes.
        Columns will contain boolean values.

        :return: None
        """

        self.labels = tf.keras.utils.to_categorical(self.labels, 6)

    def shuffle_data(self, random_state=100):
        """
        Shuffle features and labels.

        :return: None
        """
        self.features, self.labels = utils.shuffle(self.features,
                                                   self.labels,
                                                   random_state=random_state)

    def scale_features(self):
        """
        Scale the features to the range between 0 and 1.

        :return: None
        """
        self.features /= 255.0

    def class_weights(self) -> dict:
        """
        Compute class weights when you have imbalanced classes

        :return: dict
        """
        class_series = np.argmax(self.y_train, axis=1)
        class_labels = np.unique(class_series)
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=class_labels,
                                             y=class_series)
        return dict(zip(class_labels, class_weights))

    def visualise_shapes(self):
        """
        Visualise shapes of features and labels.

        :return: None
        """
        print(f"{self.features.shape=}", f"{self.labels.shape=}")
        print(f"{self.features.ndim=}", f"{self.labels.ndim=}")

    def split_to_features_and_labels(self, percentage=0.80):
        """
        Split data into train and test sets by a specified ratio.

        :param percentage: Ratio to split features and label
        :return: None
        """
        nr_of_samples = self.labels.shape[0]
        split = int(np.ceil(nr_of_samples * percentage))
        self.X_train, self.X_test = self.features[:split], self.features[split:]
        self.y_train, self.y_test = self.labels[:split], self.labels[split:]

    def split_to_validation_sets(self, percentage=0.90):  # 0.92
        """
        Split data into train and validation sets by a specified ratio.

        :param percentage: Ratio to split features and label
        :return: None
        """
        nr_of_samples = self.y_train.shape[0]
        split = int(np.ceil(nr_of_samples * percentage))
        self.X_train, self.X_valid = self.X_train[:split], self.X_train[split:]
        self.y_train, self.y_valid = self.y_train[:split], self.y_train[split:]

    def create_model(self,
                     epochs: int = 400,
                     batch_size: int = 32,
                     neurons_per_layer: list | tuple = (4200, 3000, 2000),
                     dropout: list | tuple = (0.31, 0.31, 0.4),
                     activation: list | tuple = ("relu", "relu", "relu"),
                     input_shape: int = 200):
        """
        Define DL model with all its parameters.

        :param epochs:
            Number of times that the learning algorithm will work through the entire training dataset.
        :param batch_size:
            Specify batch size to meet memory constraints, performance, convergence...
        :param neurons_per_layer:
            Number of nodes per layer.
        :param dropout:
            Percentage of nodes to kill.
        :param activation:
            Activation function to let the neuron activate or not.
        :param input_shape: Image in pixels. (height=img_size, width=img_size)

        :return: None
        """

        # DENSE NETWORK
        assert len(neurons_per_layer) == len(dropout) == len(activation), \
            "'neurons_per_layer', 'dropout' and 'activation' must be the same length" \
            "when specified in function parameters"

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=[input_shape,
                                                            input_shape,
                                                            1]))  # had to write down "1" to match the shape
        for neurons_, dropout_, activation_ in zip(neurons_per_layer,
                                                   dropout,
                                                   activation):
            self.model.add(tf.keras.layers.Dense(neurons_, activation=activation_))
            self.model.add(tf.keras.layers.Dropout(dropout_))
        self.model.add(tf.keras.layers.Dense(6, activation="softmax"))

        # CONVOLUTIONAL NETWORK
        # self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Conv2D(32,
        #                                       3,
        #                                       strides=1,
        #                                       padding="same",
        #                                       activation="relu",
        #                                       input_shape=[input_shape,
        #                                                    input_shape,
        #                                                    1]))  # had to write down "1" to match the shape
        # self.model.add(tf.keras.layers.MaxPool2D(2))
        # self.model.add(tf.keras.layers.Conv2D(32,
        #                                       3,
        #                                       strides=1,
        #                                       padding="same",
        #                                       activation="relu"))
        # self.model.add(tf.keras.layers.MaxPool2D(2))
        # self.model.add(tf.keras.layers.Flatten())
        # self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        # self.model.add(tf.keras.layers.Dropout(0.3))
        # self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        # self.model.add(tf.keras.layers.Dropout(0.3))
        # self.model.add(tf.keras.layers.Dense(6, activation="softmax"))

        self.batch_size = batch_size
        self.epochs = epochs

    def compile_model(self,
                      initial_learning_rate: float = 0.001,
                      final_learning_rate: float = 0.00001):
        """
        :param initial_learning_rate:
            Determines the step size at each iteration while moving
            toward a minimum of a loss function when the curve is still steep.
        :param final_learning_rate:
            Determines the step size at each iteration when the model reaches minimum
            of loss function.

        :return: None
        """

        # this learning schedule block was copied from net
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / self.epochs)
        steps_per_epoch = int(self.labels.shape[0] / self.batch_size)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        # categorical accuracy & loss due to one hot encoded labels
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                           metrics=[tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.CategoricalAccuracy()])

    def fit_model(self,
                  include_validation=True,
                  monitor="loss",
                  patience: int = 5):
        """

        :param include_validation:
            If true, include also validation set into fitting.
        :param monitor:
            Select function that will monitor progress of learning.
            If no progress, learning will stop by early stopping.
        :param patience:
            Limit of epochs without progress before early stopping is called.

        :return: None
        """

        callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)
        validation_data = (self.X_valid, self.y_valid) if include_validation else None
        self.history = self.model.fit(self.X_train,
                                      self.y_train,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      callbacks=callback,
                                      validation_data=validation_data,
                                      class_weight=self.class_weights())

    def save_model(self):
        """
        Save model into directory.

        :return: None
        """

        self.model.save(os.path.join(c.DL_MODELS_DIR, "trained_model.H5"))

    def visualise_models_learning(self):
        """
        Visualise model's training history with matplotlib.

        :return: None
        """
        plt.figure(figsize=(16, 9))
        plt.plot(pd.DataFrame(self.history.history))
        plt.grid(True)
        plt.show()

    def evaluate_model(self, features: np.array, labels: np.array):
        """
        Evaluate a model on unseen data.

        :return: None
        """
        result = self.model.evaluate(features, labels)
        print(f"loss: {result[0]}",
              f"precision: {result[1]}",
              f"recall: {result[2]}",
              f"categorical_accuracy: {result[3]}",
              sep="\n")
