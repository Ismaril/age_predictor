import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import constants as c
import tensorflow as tf

from data_preparation.data_transformation import DataTransformation
from PIL import Image, ImageTk, ImageOps


FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


# todo: inherit from datacleaning
class GUI(tk.Tk, DataTransformation):
    def __init__(self):
        DataTransformation.__init__(self)
        self.model = self.load_model()
        self.all_predictions = []
        self.valid_images = []
        self.index = 0

        tk.Tk.__init__(self)
        self.grid()
        self.minsize(1000, 750)
        self.title("")
        self.main_grid = tk.Frame(
            self,
            bg="Red",
            bd=3,
            border=36,
            relief="raised",
        )
        self.main_grid.pack(fill="both", expand=True)
        self.main_grid.grid_propagate(0)
        self.trash = []

    def load_model(self):
        return tf.keras.models.load_model(r"firstgood_trained_model.H5") #

    def visualise_image(self, image_name):
        # get the photo
        img = ImageTk.PhotoImage(Image.open(f"200_200_gray/{image_name}"))
        label = tk.Label(self.main_grid, image=img)
        label.place(anchor="center", relx=0.5, rely=0.5)
        self.trash.append([img, label])

    def description(self):
        # text
        text_descr = "PREDICTION"
        label = tk.Label(self.main_grid,
                         text=text_descr,
                         bg="red",
                         font=("Arial", 20))
        label.place(anchor="n", relx=0.5, rely=0.1)
        self.trash.append([label])

    def visualise_prediction(self, prediction):
        # get the predictions
        label_ranges = tk.Label(self.main_grid, text=[f" {range_} " for range_ in c.AGE_RANGES.values()],
                                bg="red",
                                font=("Arial", 20))
        label_ranges.place(anchor="n", relx=0.5, rely=0.15)

        label_predictions = tk.Label(self.main_grid,
                                     text=[f"{pred * 100:.2f}% " for pred in prediction], bg="red",
                                     font=("Arial", 20))
        label_predictions.place(anchor="n", relx=0.5, rely=0.2)

        self.trash.append([label_predictions, label_ranges])

    def buttons(self):
        button1 = tk.Button(
            self.main_grid,
            command=self.next_image_right,
            text=f">",
            height=4,
            width=15
        )
        button1.place(
            relx=0.7,
            rely=0.5,
            anchor="w",
        )
        button2 = tk.Button(
            self.main_grid,
            command=self.next_image_left,
            text=f"<",
            height=4,
            width=15,
        )
        button2.place(
            relx=0.3,
            rely=0.5,
            anchor="e",
        )

    def next_image_right(self):
        self.index += 1
        self.visual_main()

    def next_image_left(self):
        self.index -= 1
        self.visual_main()

    def destroy_widgets(self):
        for widget in self.main_grid.place_slaves():
            widget.destroy()

    def visual_main(self):
        self.destroy_widgets()
        self.description()
        self.buttons()
        self.visualise_prediction(self.all_predictions[self.index])
        self.visualise_image(self.valid_images[self.index])

    def predictions(self, path_in, image_size):
        for picture_array, img_name in zip(self.picture_array, os.listdir(path_in)):
            picture_array = picture_array / 255
            picture_array = np.ndarray.reshape(picture_array, (1, image_size, image_size))
            prediction = np.ndarray.flatten(self.model.predict(picture_array))
            self.all_predictions.append(prediction)
            self.valid_images.append(img_name)


