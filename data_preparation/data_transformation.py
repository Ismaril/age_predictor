import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance, ImageOps


class DataTransformation:
    def __init__(self):
        self.labels_prepared = []
        self.picture_array = []

    @staticmethod
    def extract_faces(path_in: str,
                      path_out: str,
                      prefix: str = "",
                      img_size=200,
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

        :param path_in:
            Location of source images.
        :param path_out:
            Location where to save new images.
        :param img_size:
            Desired size of image in pixels. (height=img_size, width=img_size)
        :param scale_factor:
            Specifying how much the image size is reduced at each image scale. Meaning
            how many times closer will the face be in the final image.
        :param min_neighbours:
            Specifying how many neighbors each candidate rectangle should have to retain it.
            The higher the number, the longer it will take to extract the faces, the better
            (i guess) the extraction should be...
        :param prefix: Add a prefix to a file name.

        :return: None
        """

        # Classifier algorithm.
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        # Locate face and save img for each image in a given dir.
        for image_name in os.listdir(path_in):

            # Array of [R, G, B] for each pixel.
            img = cv2.imread(os.path.join(path_in, image_name))

            # Convert to grayscale. (better for object detection)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # returns 4 points that are a corners that detected face (rectangle area around face)
            faces = face_cascade.detectMultiScale(gray,
                                                  scaleFactor=scale_factor,
                                                  minNeighbors=min_neighbours)

            # Cutting actual face from original img.
            # (...by setting only relevant pixels based on input from "detectMultiscale"?)
            try:
                for (x, y, w, h) in faces:
                    roi_color = img[y:y + h, x:x + w]

                # Resize to n by n pixels.
                resized = cv2.resize(roi_color, (img_size, img_size))

                # Save
                cv2.imwrite(f"{path_out}/{prefix}{image_name}", resized)

            # TODO: specify exact except condition
            except:
                print("No faces detected")

        print("Face extraction - done")

    # TODO: this function can be deleted once I locate the bug
    @staticmethod
    def remove_duplicates(path: str):
        """
        Remove duplicate images from a given folder.

        :return: None
        """
        data = os.listdir(path)

        # Locate duplicates - check for file equality and return file name.
        # This will only locate equal images if they are all the same in a row.
        # Checking for the same image randomly in the whole folder would require
        #   different implementation here.
        to_be_deleted = []
        for i, image_name in enumerate(data[:-1]):
            with Image.open(os.path.join(path, image_name)) as img1:
                with Image.open(os.path.join(path, data[i + 1])) as img2:
                    if img1 == img2:
                        to_be_deleted.append(data[i + 1])

        # Remove duplicates.
        for image_name in data:
            if image_name in to_be_deleted:
                os.remove(os.path.join(path, image_name))

        print("Duplicate removal - done")

    @staticmethod
    def change_brightness(path,
                          prefix1="drk_",
                          factor1=0.9,
                          prefix2="brt_",
                          factor2=1.1):
        """
        Data augmentation function that's gonna inflate feature numbers by 200%.

        :param path:
            1. Is the path were your files are located.
            2. Is the output directory at the same time.
        :param prefix1:
            Add a prefix to enhanced image's name.
        :param factor1:
            For darker image use number smaller than 1,
            for brighter number higher than 1.
        :param factor2:
            see factor1
        :param prefix2:
            see prefix1

        :return: None
        """

        for image_name in os.listdir(path):
            # Read the image.
            image_obj = Image.open(os.path.join(path, image_name))

            # Image brightness enhancer.
            enhancer = ImageEnhance.Brightness(image_obj)

            # Brightness is changed here.
            im_output1 = enhancer.enhance(factor1)
            im_output2 = enhancer.enhance(factor2)

            # Save images.
            im_output1.save(os.path.join(path, f"{prefix1}{image_name}"))
            im_output2.save(os.path.join(path, f"{prefix2}{image_name}"))

        print("Brightness adjustment - done")

    @staticmethod
    def convert_to_grayscale(path_in: str,
                             path_out: str):
        """
        Convert images to grayscale.

        :param path_in: Input directory of colored images.
        :param path_out: Output directory for grayscale images.

        :return: None
        """

        for image_name in os.listdir(path_in):
            with Image.open(os.path.join(path_in, image_name)) as img:
                gray_scaled = ImageOps.grayscale(img)
                gray_scaled.save(os.path.join(path_out, image_name))

        print("Grayscale conversion - done")

    @staticmethod
    def balance_classes(path_in: str,
                        path_out: str,
                        limit: int = 1000):
        """

        :param path_in: Input directory.
        :param path_out: Output directory.
        :param limit: Specify how many features each class should have.

        :return: None
        """

        all_files = os.listdir(path_in)
        difference = len(all_files) - limit

        random.shuffle(all_files)

        for i, image_name in enumerate(all_files):
            if i == difference or len(all_files) < limit:
                break
            else:
                shutil.move(os.path.join(path_in, image_name),
                            os.path.join(path_out, image_name))

        print("Classes balanced - done")

    def extract_labels(self, path_in):
        """
        Extract ages from the image name and assign labels.

        20-27 -> 0,
        28-35 -> 1,
        36-44 -> 2,
        45-55 -> 3,
        56-65 -> 4,
        66<=  -> 5

        :param path_in: Input directory.
        :return: None
        """

        for image_name in os.listdir(path_in):
            split = int(image_name.split(".")[0][-2:])

            if split in range(20, 28):
                self.labels_prepared.append(0)
            elif split in range(28, 36):
                self.labels_prepared.append(1)
            elif split in range(36, 45):
                self.labels_prepared.append(2)
            elif split in range(45, 56):
                self.labels_prepared.append(3)
            elif split in range(56, 66):
                self.labels_prepared.append(4)
            elif split in range(66, 120):
                self.labels_prepared.append(5)

        print("Label extraction - done ")

    def save_labels(self, path_out):
        """
        Save all labels as a csv dataset.

        :param path_out: Output directory.
        :return: None
        """
        labels = pd.DataFrame(self.labels_prepared, columns=["labels"])
        labels.to_csv(os.path.join(path_out, "labels_prepared.csv"))

        print("Labels saved - done")

    def convert_to_array(self, path_in):
        """
        Convert images to numpy arrays.

        :param path_in: Input directory.
        :return: None
        """

        for image_name in os.listdir(path_in):
            image = Image.open(os.path.join(path_in, image_name))
            self.picture_array.append(np.array(image))

        print("Conversion of photos to arrays - done")

    def save_array(self, path_out):
        """
        Save all images in numpy arrays as *.npy file.

        :param path_out: Output directory.
        :return: None
        """

        array_ = np.array(self.picture_array)
        np.save(os.path.join(path_out, "picture_array.npy"), array_)

        print("Arrays saved - done")
