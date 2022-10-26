import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import utils


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
        Load features and labels

        :return: None
        """
        assert path_features.endswith(".npy"), "File must be *.npy"
        assert path_labels.endswith(".csv"), "File must be *.csv"

        features = np.load(path_features)
        labels = pd.read_table(path_labels, delimiter=",")
        labels = np.array(labels["label"])

        self.features = features
        self.labels = labels

    def change_dtypes(self):
        """
        Change data types of features and labels to float

        :return: None
        """

        self.features = self.features.astype(float)
        self.labels = self.labels.astype(float)

    def one_hot_encode_labels(self):
        """
        Set number of columns based on number of classes that will contain boolean values.

        :return: None
        """
        self.labels = tf.keras.utils.to_categorical(self.labels, 6)

    def shuffle_data(self, random_state=100):
        """
        Shuffle features and labels

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

    def visualise_shapes(self):
        """
        Visualise shapes of features and labels

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
                     epochs: int = 400,  # best 300
                     batch_size: int = 32,
                     neurons_per_layer: list | tuple = (4200, 3000, 2000),
                     dropout: list | tuple = (0.31, 0.31, 0.4),
                     activation: list | tuple = ("relu", "relu", "relu")):  # best 32
        """
        Define DL model with all its parameters.

        :param epochs:
            number times that the learning algorithm will work through the entire training dataset
        :param batch_size:
            The bigger the batch size, the faster the result, the worse the performance
            The smaller the batch size, the longer time to train, the better the result
            Still batch under 32 does not make sense, according to tests,
                because time hugely increases and results are more or less the same.
        :param neurons_per_layer: Number of nodes per layer
        :param dropout: % of nodes to kill
        :param activation: activation function to let the neuron activate or not
        :return: None
        """

        assert len(neurons_per_layer) == len(dropout) == len(activation), \
            "'neurons_per_layer', 'dropout' and 'activation' must be the same length"

        # todo: vyzkoušet o chlup větší regularizaci třeba 0.315 nebo 0.32
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=[200, 200]))
        for neurons_, dropout_, activation_ in zip(neurons_per_layer,
                                                   dropout,
                                                   activation):
            self.model.add(tf.keras.layers.Dense(neurons_, activation=activation_))
            self.model.add(tf.keras.layers.Dropout(dropout_))
        self.model.add(tf.keras.layers.Dense(6, activation="softmax"))

        self.batch_size = batch_size
        self.epochs = epochs

    def compile_model(self,
                      initial_learning_rate: float = 0.001,
                      final_learning_rate: float = 0.00001):
        """
        :param initial_learning_rate:
            determines the step size at each iteration while moving toward a minimum of a loss function
                at the beginning of all epochs
        :param final_learning_rate:
            determines the step size at each iteration while moving toward a minimum of a loss function
                at the end of all epochs
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

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                           metrics=[tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tf.keras.metrics.CategoricalAccuracy()])  # categoricalaccuracy due to one hot encoded labels

    def fit_model(self,
                  include_validation=True,
                  monitor="loss",
                  patience: int = 5):
        """

        :param include_validation: if true, include also validation set into fitting
        :param monitor:
            select function that will monitor progress of learning. If no progress,
                learning will stop by early stopping
        :param patience: limit of epochs without progress before early stopping is called
        :return:
        """

        callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)
        validation_data = (self.X_valid, self.y_valid) if include_validation else None

        self.history = self.model.fit(self.X_train,
                                      self.y_train,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      callbacks=callback,
                                      validation_data=validation_data)

    def save_model(self, path_name: str):
        """
        Save model into specified directory
        @param path_name: Insert path/name_of_file.H5 to save the model.
        @return: None
        """
        assert path_name.endswith(".H5"), "Save model with extension *.H5"
        self.model.save("trained_model.H5")

    def visualise_models_learning(self):
        # todo: set labels to all evaluation metrcis and change colors of lines
        # plt.style.use("dark_background")
        plt.figure(figsize=(16, 9))
        plt.plot(pd.DataFrame(self.history.history))
        plt.grid(True)
        # plt.gca().set_ylim(0, 1)  # set vertical range
        plt.show()


# OPERATE HERE #################################################################################
model1 = ModelTraining()
model1.load_data(path_features="C:/Users/lazni/PycharmProjects/Age_Predictor/picture_array.npy",
                 path_labels="C:/Users/lazni/PycharmProjects/Age_Predictor/labels_prepared.csv")
model1.change_dtypes()
model1.one_hot_encode_labels()
model1.shuffle_data(random_state=100)
model1.scale_features()
model1.visualise_shapes()
model1.split_to_features_and_labels(percentage=0.8)
model1.split_to_validation_sets(percentage=0.9)
model1.create_model(epochs=5,
                    batch_size=32,
                    neurons_per_layer=(4200, 3000, 2000),
                    dropout=(0.31, 0.31, 0.4),
                    activation=("relu", "relu", "relu"))
model1.compile_model(initial_learning_rate=0.001,
                     final_learning_rate=0.00001)
model1.fit_model(include_validation=True,
                 monitor="loss",
                 patience=5)
model1.save_model("trained_model.H5")
model1.visualise_models_learning()

# C:\Users\lazni\AppData\Local\Temp\CUDA


# 385
# 185
#
# 400
#
# acc 1.000
# loss 0.0159
# acc val 92.49
# val loss 0.3380
