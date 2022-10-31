import os
import random
import shutil
import time
import cv2
import numpy as np
import pandas as pd
import constants as c
from PIL import Image, ImageEnhance, ImageOps


class DataTransformation:
    def __init__(self):
        self.labels_original = []
        self.labels_prepared = []
        self.picture_array = []

    @staticmethod
    def extract_faces(path_in: str,
                      path_out: str,
                      prefix: str = "",
                      img_size=128,
                      scale_factor=1.3,
                      min_neighbours=3):

        # TODO: find out why some faces are duplicated when algorithm cannot locate a
        #   face in a given iteration

        """
        Return cut faces from raw images

        Pycharm has some troubles with search of all methods and classes form cv2,
        but code works

        Code by:
        1. https://github.com/opencv/opencv/tree/4.x/data/haarcascades/haarcascade_frontalface_default.xml'
        2. Some opencv documentation

        :param path_in: location of source images
        :param path_out: location where to save new images
        :param img_size: desired size of image in pixels
        :param scale_factor:
            specifying how much the image size is reduced at each image scale. Meaning
            how many times closer will the face be in the final image.
        :param min_neighbours:
            specifying how many neighbors each candidate rectangle should have to retain it.
            the higher the number, the longer it will take to extract the faces, the better
            the extraction should be
        :param prefix: add a prefix to a file name

        """

        # classifier algorithm
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        # locate face and save img for each image in a given dir
        for image_name in os.listdir(path_in):

            # array of [R, G, B] for each pixel
            img = cv2.imread(os.path.join(path_in, image_name))

            # convert to grayscale (better for object detection)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # returns 4 points that are a corners that detected face (rectangle area around face)
            faces = face_cascade.detectMultiScale(gray,
                                                  scaleFactor=scale_factor,
                                                  minNeighbors=min_neighbours)

            # cutting actual face from original img, ...by setting only relevant pixels?
            try:
                for (x, y, w, h) in faces:
                    roi_color = img[y:y + h, x:x + w]

                # resize to n by n pixels
                resized = cv2.resize(roi_color, (img_size, img_size))

                cv2.imwrite(f"{path_out}/{prefix}{image_name}", resized)

            except:  # TODO: specify exact except condition
                print("No faces detected")

        print("Face extraction - done")

    # TODO: this function can be deleted once I locate the bug
    @staticmethod
    def remove_duplicates(path: str):
        """
        Remove duplicate images once they were extracted by opencv2.

        """
        data = os.listdir(path)

        # locate duplicates (check for file equality and return file name)
        to_be_deleted = []
        for i, image_name in enumerate(data[:-1]):
            with Image.open(os.path.join(path, image_name)) as img1:
                with Image.open(os.path.join(path, data[i + 1])) as img2:
                    if img1 == img2:
                        to_be_deleted.append(data[i + 1])

        # remove duplicates
        for image_name in data:
            if image_name in to_be_deleted:
                os.remove(os.path.join(path, image_name))
        print("Duplicate removal - done")

    def change_brightness(self,
                          path,
                          prefix1="drk_",
                          factor1=0.9,
                          prefix2="brt_",
                          factor2=1.1):
        """
        Data augmentation function that is gonna inflate feature numbers (picture dataset)
        by 200%.

        :param path:
            1. is the path were your files are located
            2. is the output directory at the same time
        :param prefix1: add a new name (only prefix) to enhanced image
        :param factor1: for darker image use number smaller than 1,
            for brighter number higher than 1
        :param factor2: see factor1
        :param prefix2: see prefix1
        """

        for image_name in os.listdir(path):
            # read the image
            image_obj = Image.open(os.path.join(path, image_name))

            # image brightness enhancer
            enhancer = ImageEnhance.Brightness(image_obj)

            im_output1 = enhancer.enhance(factor1)  # brightness changed here
            im_output2 = enhancer.enhance(factor2)  # brightness changed here

            im_output1.save(os.path.join(path, f"{prefix1}{image_name}"))
            im_output2.save(os.path.join(path, f"{prefix2}{image_name}"))

        print("Brightness adjustment - done")

    @staticmethod
    def convert_to_grayscale(path_in: str,
                             path_out: str):
        """
        Convert images to grayscale

        :param path_in: link to the directory of colored images
        :param path_out: link to the directory for grayscale images
        :return Image:
        """

        for image_name in os.listdir(path_in):
            with Image.open(os.path.join(path_in, image_name)) as img:
                gray_scaled = ImageOps.grayscale(img)
                gray_scaled.save(os.path.join(path_out, image_name))

        print("Grayscale conversion - done")

    def balance_classes(self, path_in: str, path_out: str, limit: int = 1000):

        all_files = os.listdir(path_in)
        difference = len(all_files) - limit

        random.shuffle(all_files)

        for i, image_name in enumerate(all_files):
            if i == difference or i < 0:
                break
            else:
                shutil.move(os.path.join(path_in, image_name),
                            os.path.join(path_out, image_name))

        print("Classes balanced - done")

    def extract_labels(self, path_in):
        # locate old labels from the file name
        for image_name in os.listdir(path_in):
            self.labels_original.append(image_name.split(".")[0][-2:])

        print("Label extraction - done ")

    def save_labels(self, path_out):
        """
        assign new labels based on category

        20-27 -> 0,
        28-35 -> 1,
        36-44 -> 2,
        45-55 -> 3,
        56-65 -> 4,
        66<=  -> 5
        """

        for label in self.labels_original:
            label = int(label)
            if label in range(20, 28):
                self.labels_prepared.append(0)
            elif label in range(28, 36):
                self.labels_prepared.append(1)
            elif label in range(36, 45):
                self.labels_prepared.append(2)
            elif label in range(45, 56):
                self.labels_prepared.append(3)
            elif label in range(56, 66):
                self.labels_prepared.append(4)
            elif label in range(66, 120):
                self.labels_prepared.append(5)

        labels = pd.DataFrame(self.labels_prepared, columns=["labels"])
        labels.to_csv(os.path.join(path_out, "labels_prepared.csv"))

        print("Labels saved - done")

    def convert_to_array(self, path_in):
        """
        Convert images to arrays.
        """

        for image_name in os.listdir(path_in):
            image = Image.open(os.path.join(path_in, image_name))
            self.picture_array.append(np.array(image))

        print("Conversion of photos to arrays - done")

    def save_array(self, path_out):
        array_ = np.array(self.picture_array)
        np.save(os.path.join(path_out, "picture_array.npy"), array_)

        print("Arrays saved - done")


#################################################################################
# OPERATE HERE

start = time.perf_counter()
transformer = DataTransformation()

for folder in os.listdir(c.IMAGES_RAW_DIR):
    current_input = os.path.join(c.IMAGES_RAW_DIR, folder)
    current_output = os.path.join(c.IMAGES_TRANSFORMED_DIR, folder)

    transformer.extract_faces(scale_factor=1.01,
                              min_neighbours=70,
                              prefix="o_",
                              img_size=128,
                              path_in=current_input,
                              path_out=current_output)
    transformer.remove_duplicates(current_output)

    transformer.extract_faces(scale_factor=1.3,
                              prefix="13_",
                              min_neighbours=8,
                              img_size=128,
                              path_in=current_input,
                              path_out=current_output)
    transformer.remove_duplicates(current_output)

    transformer.change_brightness(path=current_output,
                                  prefix1="drk_",
                                  factor1=0.9,
                                  prefix2="brt_",
                                  factor2=1.1)
    transformer.convert_to_grayscale(path_in=current_output,
                                     path_out=current_output)
    transformer.balance_classes(path_in=current_output,
                                path_out=os.path.join(c.IMAGES_ADDITIONAL_DIR, folder),
                                limit=6080)
    transformer.extract_labels(current_output)
    transformer.convert_to_array(current_output)
    print(f"{folder} - done", "_"*60, sep="\n")

transformer.save_labels(c.PROJECT_PARENT_DIR)
transformer.save_array(c.PROJECT_PARENT_DIR)

end = time.perf_counter()
print(f"Data transformation completed in: \n{end - start} seconds")
