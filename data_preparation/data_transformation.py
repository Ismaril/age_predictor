import os
import time

from PIL import Image
from PIL import ImageOps
import cv2


def convert_to_grayscale(path_in, path_out):
    """
    Convert images to grayscale
    :param path_in: link to the directory of colored images
    :param path_out: link to the directory for grayscale images
    :return Image:
    """
    for image in os.listdir(path_in):
        with Image.open(f"{path_in}/{image}") as img:
            gray_scaled = ImageOps.grayscale(img)
            gray_scaled.save(f"{path_out}/{image}")
    print("Grayscale conversion - done")


def extract_labels(path_in, path_out_labels):
    """
    Extract labels and convert them to ML format
    20-27 -> 0,
    28-35 -> 1,
    36-44 -> 2,
    45-55 -> 3,
    56-65 -> 4,
    66<=  -> 5
    """

    # locate old labels from the file name
    labels_original = []
    for person in os.listdir(path_in):
        labels_original.append(person[-6:-4])

    # assign new labels based on category
    with open(f"{path_out_labels}/labels_prepared.csv", "w") as file:
        file.write("label,\n")
        for label in labels_original:
            label = int(label)
            if label in range(20, 28):
                file.write("0,\n")
            elif label in range(28, 36):
                file.write("1,\n")
            elif label in range(36, 45):
                file.write("2,\n")
            elif label in range(45, 56):
                file.write("3,\n")
            elif label in range(56, 66):
                file.write("4,\n")
            elif label in range(66, 120):
                file.write("5,\n")
    print("Label extraction - done")


def extract_faces(path_in,
                  path_out,
                  img_size=128,
                  scale_factor=1.3,
                  min_neighbours=2):
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
    :param scale_factor: tweak ML algorithm which detects faces
    :param min_neighbours: tweak ML algorithm which detects faces

    """

    # classifier algorithm
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # locate face and save img for each image in a given dir
    for image in os.listdir(path_in):
        img = cv2.imread(f"{path_in}/{image}")  # array of [R, G, B] for each pixel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale (better for object detection)
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=scale_factor,

                                              # probably returns 4 points that are a corners
                                              # that detected face (rectangle area around face)
                                              minNeighbors=min_neighbours)
        try:
            # cutting actual face from original img by setting only relevant pixels?
            for (x, y, w, h) in faces:
                roi_color = img[y:y + h, x:x + w]

            resized = cv2.resize(roi_color, (img_size, img_size))  # resize to n by n pixels
            cv2.imwrite(f"{path_out}/{image}", resized)
        except:
            print("No faces detected")
    print("Face extraction done")


def remove_duplicates(path):
    data = os.listdir(path)

    # locate duplicates (check for file equality and return file name)
    to_be_deleted = []
    for i, image in enumerate(data[:-1]):
        with Image.open(f"{path}/{image}") as img1:
            with Image.open(f"{path}/{data[i + 1]}") as img2:
                if img1 == img2: to_be_deleted.append(data[i + 1])

    # remove duplicates
    for file in data:
        if file in to_be_deleted:
            os.remove(f"{path}/{file}")
    print("Duplicate removal - done")


#################################################################################
def main(path_in, path_out_images, path_out_labels):
    start = time.perf_counter()
    extract_faces(scale_factor=1.4,
                  min_neighbours=3,
                  img_size=128,
                  path_in=path_in,
                  path_out=path_out_images)
    remove_duplicates(path=path_out_images)
    convert_to_grayscale(path_in=path_out_images, path_out=path_out_images)
    extract_labels(path_in=path_out_images, path_out_labels=path_out_labels)
    end = time.perf_counter()
    print(f"Data transformation completed in: {end - start} seconds")


in_ = "C:/Users/lazni/PycharmProjects/Age_Predictor/images/label_0"
out_img = "C:/Users/lazni/PycharmProjects/Age_Predictor/images/transformed"
out_labels = "C:/Users/lazni/PycharmProjects/Age_Predictor"

main(path_in=in_, path_out_images=out_img, path_out_labels=out_labels)
