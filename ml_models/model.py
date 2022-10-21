from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import utils

# load data
features = np.load("C:/Users/lazni/PycharmProjects/Age_Predictor/picture_array.npy")
labels = pd.read_table("C:/Users/lazni/PycharmProjects/Age_Predictor/labels_prepared.csv",
                       delimiter=",")
labels = np.array(labels["label"])

# change dtypes
features = features.astype(float)
labels = labels.astype(float)

# shuffle features and labels
# (each feature will still match with label after shuffling...)
features, labels = utils.shuffle(features, labels, random_state=100)

# scale the features to the range between 0 and 1
features /= 255.0

# visualise shapes
print(features.shape, labels.shape)


def split_to_features_and_labels(features,
                                 labels,
                                 percentage=0.8):
    """
    split data into train and test
    """
    nr_of_samples = labels.shape
    nr_of_samples = nr_of_samples[0]
    split = int(np.ceil(nr_of_samples * percentage))
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = labels[:split], labels[split:]
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = split_to_features_and_labels(features=features,
                                                                labels=labels)


def split_to_validation_sets(X_train_,
                             y_train_,
                             percentage=0.92):
    nr_of_samples = y_train_.shape
    nr_of_samples = nr_of_samples[0]
    split = int(np.ceil(nr_of_samples * percentage))
    X_train_, X_valid = X_train_[:split], X_train_[split:]
    y_train_, y_valid = y_train_[:split], y_train_[split:]
    return X_train_, X_valid, y_train_, y_valid


X_train, X_valid, y_train, y_valid = split_to_validation_sets(X_train_=X_train,
                                                              y_train_=y_train)

# ### train model ###
# create the model
# todo: model is underfitting with current parameters
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[200, 200]))
model.add(tf.keras.layers.Dense(250, activation="relu"))  # best 385, relu
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(200, activation="relu"))  # best 185, relu
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(6, activation="softmax"))

epochs = 400  # best 300
# todo: zkusit batch 16
batch_size = 32  # best 32

# compile the model
# todo: check momentum and decay
initial_learning_rate = 0.003  # best 0.003
final_learning_rate = 0.000001  # best 0.00001
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
steps_per_epoch = int(labels.shape[0] / batch_size)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=steps_per_epoch,
    decay_rate=learning_rate_decay_factor,
    staircase=True)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              metrics=["accuracy"])

# fit the model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3)

history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    # callbacks=callback,
                    validation_data=(X_valid, y_valid))

# visualise
# plt.style.use("dark_background")
plt.figure(figsize=(16, 9))
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
# plt.gca().set_ylim(0, 1)  # set vertical range
plt.show()

# save model
model.save("trained_model.H5")

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
