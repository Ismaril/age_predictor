import os
import numpy as np
import tkinter as tk
import constants as c
import tensorflow as tf

from PIL import Image, ImageTk
from data_preparation.data_transformation import DataTransformation


class GUI(tk.Tk, DataTransformation):
    def __init__(self):
        DataTransformation.__init__(self)
        tk.Tk.__init__(self)

        self.model = self.__load_model()

        self.all_predictions = []
        self.valid_images = []
        self.index = 0

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

    @staticmethod
    def __load_model() -> tf.keras.models.Model:
        return tf.keras.models.load_model(r"best_model.H5")

    def __visualise_image(self, image_name):
        img = ImageTk.PhotoImage(Image.open(f"transformed/{image_name}"))
        label = tk.Label(self.main_grid,
                         image=img)
        label.place(anchor="center",
                    relx=0.5,
                    rely=0.5)
        self.trash.append([img, label])

    def __description(self):
        text_descr = "PREDICTION"
        label = tk.Label(self.main_grid,
                         text=text_descr,
                         bg="red",
                         font=("Arial", 20))
        label.place(anchor="n",
                    relx=0.5,
                    rely=0.1)
        self.trash.append([label])

    def __visualise_prediction(self, prediction):
        label_ranges = tk.Label(self.main_grid,
                                text=[f" {range_} " for range_ in c.AGE_RANGES.values()],
                                bg="red",
                                font=("Arial", 20))
        label_ranges.place(anchor="n",
                           relx=0.5,
                           rely=0.15)
        label_predictions = tk.Label(self.main_grid,
                                     text=[f"{pred * 100:.2f}% " for pred in prediction],
                                     bg="red",
                                     font=("Arial", 20))
        label_predictions.place(anchor="n",
                                relx=0.5,
                                rely=0.2)

        self.trash.append([label_predictions, label_ranges])

    def __visualise_name(self, image_name):
        label = tk.Label(self.main_grid,
                         text=image_name,
                         bg="red",
                         font=("Arial", 20))
        label.place(anchor="n",
                    relx=0.5,
                    rely=0.25)
        self.trash.append(label)

    def __next_image_right(self):
        self.index += 1

        if abs(self.index) == len(self.valid_images):
            self.index = 0

        self.visual_main()

    def __next_image_left(self):
        self.index -= 1

        if abs(self.index) == len(self.valid_images):
            self.index = 0

        self.visual_main()

    def __buttons(self):
        button1 = tk.Button(
            self.main_grid,
            command=self.__next_image_right,
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
            command=self.__next_image_left,
            text=f"<",
            height=4,
            width=15,
        )
        button2.place(
            relx=0.3,
            rely=0.5,
            anchor="e",
        )

    def __destroy_widgets(self):
        for widget in self.main_grid.place_slaves():
            widget.destroy()

    @staticmethod
    def images_present_raw_dir():
        assert os.listdir(c.PRODUCTION_RAW_DIR), "You did not input any images into raw_images directory."

    @staticmethod
    def images_present_transformed_dir():
        assert os.listdir(c.PRODUCTION_TRANSFORMED_DIR), "Images you took were not recognised."

    def predictions(self, path_in, image_size):
        """
        Get predictions on images from specified folder.

        :param path_in: Folder containing images.
        :param image_size: Image in pixels. (height=img_size, width=img_size)
        :return: None
        """
        for picture_array, img_name in zip(self.picture_array, os.listdir(path_in)):
            picture_array = picture_array / 255
            picture_array = np.ndarray.reshape(picture_array, (1, image_size, image_size))
            prediction = np.ndarray.flatten(self.model.predict(picture_array))
            self.all_predictions.append(prediction)
            self.valid_images.append(img_name)

    def visual_main(self):
        """Visualise all widgets"""
        self.__destroy_widgets()
        self.__description()
        self.__buttons()
        self.__visualise_prediction(self.all_predictions[self.index])
        self.__visualise_name(self.valid_images[self.index])
        self.__visualise_image(self.valid_images[self.index])
