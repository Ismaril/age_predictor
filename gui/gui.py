import os
import cv2
import numpy as np
import tkinter as tk
import tensorflow as tf
from PIL import Image, ImageTk, ImageOps

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


class GUI(tk.Tk):
    def __init__(self):
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

    def extract_faces(self):
        for image_name in os.listdir("raw_images"):
            # array of [R, G, B] for each pixel
            image = cv2.imread(f"raw_images/{image_name}")

            # convert to grayscale (better for object detection)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # returns 4 points that are a corners that detected face (rectangle area around face)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

            # cutting actual face from original img
            for (x, y, w, h) in faces:
                roi_color = image[y:y + h, x:x + w]

            # resize to n by n pixels
            resized = cv2.resize(roi_color, (200, 200))

            # save img
            cv2.imwrite(f"200_200_color/{image_name}", resized)

    def convert_to_gray_scale(self):
        for image_name in os.listdir("200_200_color"):
            image = Image.open(f"200_200_color/{image_name}")
            image_gray = ImageOps.grayscale(image)
            image_gray.save(f"200_200_gray/{image_name}")

    def load_model(self):
        return tf.keras.models.load_model(r"firstgood_trained_model.H5")

    def age_ranges(self):
        return {0: "20-27",
                1: "28-35",
                2: "36-44",
                3: "45-55",
                4: "56-65",
                5: "66<="}

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
        label_ranges = tk.Label(self.main_grid, text=[f" {range_} " for range_ in self.age_ranges().values()],
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


    def remove_duplicates(self):
        # TODO: this function can be deleted once I locate the bug
        """
        Remove duplicate images once they were extracted by opencv2.
        """
        directory = "200_200_gray"
        data = os.listdir(f"{directory}")

        # locate duplicates (check for file equality and return file name)
        to_be_deleted = []
        for i, image in enumerate(data[:-1]):
            with Image.open(f"{directory}/{image}") as img1:
                with Image.open(f"{directory}/{data[i + 1]}") as img2:
                    if img1 == img2: to_be_deleted.append(data[i + 1])

        # remove duplicates
        for file in data:
            if file in to_be_deleted:
                os.remove(f"{directory}/{file}")

    def visual_main(self):
        self.destroy_widgets()
        self.description()
        self.buttons()
        self.visualise_prediction(gui.all_predictions[self.index])
        self.visualise_image(gui.valid_images[self.index])

    def main(self):
        for image_name in os.listdir("200_200_gray"):
            image = np.array(Image.open(f"200_200_gray/{image_name}"))
            print(image_name)
            picture_array = image / 255
            picture_array = np.ndarray.reshape(picture_array, (1, 200, 200))
            prediction = np.ndarray.flatten(self.model.predict(picture_array))
            self.all_predictions.append(prediction)
            self.valid_images.append(image_name)

# todo: make sure we do not get to the end of a list with arrows
gui = GUI()
gui.extract_faces()
gui.convert_to_gray_scale()
gui.remove_duplicates()
gui.main()
gui.visual_main()
gui.mainloop()
