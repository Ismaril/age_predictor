import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

image = Image.open("C:/Users/lazni/Desktop/langos.jpg")
image_gray = ImageOps.grayscale(image)
array = np.array(image_gray) / 255.0
array = np.ndarray.reshape(array, (1, 200, 200))
print(array.shape)
print(array.ndim)

label = np.array([0])
print(label.ndim)

model = tf.keras.models.load_model("trained_model.H5")
model: tf.keras.Model
print(model.predict(array))
prediction = np.ndarray.flatten(model.predict(array))
ranges = {0: "20-27",
          1: "28-35",
          2: "36-44",
          3: "45-55",
          4: "56-65",
          5: "66<="}

for prediction, range in zip(prediction, ranges.values()):
    print(f"{prediction*100:.2f}%", range)
