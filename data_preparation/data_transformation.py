import os
import time
import cv2
import numpy as np

from PIL import Image, ImageEnhance, ImageOps


def convert_to_grayscale(path_in, path_out):
    """
    Convert images to grayscale

    :param path_in: link to the directory of colored images
    :param path_out: link to the directory for grayscale images
    :return Image:
    """
    for directory in os.listdir(path_in):
        for image in os.listdir(f"{path_in}/{directory}"):
            with Image.open(f"{path_in}/{directory}/{image}") as img:
                gray_scaled = ImageOps.grayscale(img)
                gray_scaled.save(f"{path_out}/{directory}/{image}")
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
    for directory in os.listdir(path_in):
        for person in os.listdir(f"{path_in}/{directory}"):
            labels_original.append(person[-6:-4])  # todo: split by dot

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
                  min_neighbours=2,
                  data_augmentation=False):
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
    :param data_augmentation: If True, all faces from original dataset, but with different
        zoom. (Create more features by data augmentation)

    """

    # classifier algorithm
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    for directory in os.listdir(path_in):
        if directory.startswith("label"):

            # locate face and save img for each image in a given dir
            for image in os.listdir(f"{path_in}/{directory}"):

                # array of [R, G, B] for each pixel
                img = cv2.imread(f"{path_in}/{directory}/{image}")

                # convert to grayscale (better for object detection)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # returns 4 points that are a corners that detected face (rectangle area around face)
                faces = face_cascade.detectMultiScale(gray,
                                                      scaleFactor=scale_factor,
                                                      minNeighbors=min_neighbours)

                # cutting actual face from original img by setting only relevant pixels?
                try:
                    for (x, y, w, h) in faces:
                        roi_color = img[y:y + h, x:x + w]

                    # resize to n by n pixels
                    resized = cv2.resize(roi_color, (img_size, img_size))

                    if not data_augmentation:
                        cv2.imwrite(f"{path_out}/{directory}/{image}", resized)
                    else:
                        cv2.imwrite(f"{path_out}/{directory}/_{image}", resized)

                # TODO: specify exact except condition
                except:
                    print("No faces detected")
    print("Face extraction - done")


def change_brightness(path_in,
                      prefix1="drk_",
                      factor1=0.9,
                      prefix2="brt_",
                      factor2=1.1):
    """
    Data augmentation function that is gonna inflate feature numbers (picture dataset)
    by 200%.

    :param path_in:
        1. is the path were your files are located
        2. is the output directory at the same time
    :param prefix1: add a new name (only prefix) to enhanced image
    :param factor1: for darker image use number smaller than 1,
        for brighter number higher than 1
    :param factor2: see factor1
    :param prefix2: see prefix1
    """

    for directory in os.listdir(path_in):
        for image_ in os.listdir(f"{path_in}/{directory}"):
            # read the image
            image = Image.open(f"{path_in}/{directory}/{image_}")

            # image brightness enhancer
            enhancer = ImageEnhance.Brightness(image)

            im_output1 = enhancer.enhance(factor1)  # brightness changed here
            im_output2 = enhancer.enhance(factor2)  # brightness changed here

            im_output1.save(f"{path_in}/{directory}/{prefix1}{image_}")
            im_output2.save(f"{path_in}/{directory}/{prefix2}{image_}")
    print("Brightness adjustment - done")


def remove_duplicates(path):
    # TODO: this function can be deleted once I locate the bug
    """
    Remove duplicate images once they were extracted by opencv2.
    """
    for directory in os.listdir(path):
        data = os.listdir(f"{path}/{directory}")

        # locate duplicates (check for file equality and return file name)
        to_be_deleted = []
        for i, image in enumerate(data[:-1]):
            with Image.open(f"{path}/{directory}/{image}") as img1:
                with Image.open(f"{path}/{directory}/{data[i + 1]}") as img2:
                    if img1 == img2: to_be_deleted.append(data[i + 1])

        # remove duplicates
        for file in data:
            if file in to_be_deleted:
                os.remove(f"{path}/{directory}/{file}")
    print("Duplicate removal - done")


def convert_to_array(path_in, path_out, file_name="picture_array"):
    """
    Convert images to list arrays
    """
    array_ = []
    for directory in os.listdir(path_in):
        for image in os.listdir(f"{path_in}/{directory}"):
            array_.append(np.array(Image.open(f"{path_in}/{directory}/{image}")))

    array_ = np.array(array_)
    np.save(path_out + "/" + file_name, array_)
    print("Conversion of photos to arrays - done")


#################################################################################
# OPERATE HERE
# todo: set only relative paths
# todo: try to change iteration through directories to global level
in_raw_imgs = "C:/Users/lazni/PycharmProjects/Age_Predictor/images/raw"
out_img = "C:/Users/lazni/PycharmProjects/Age_Predictor/images/transformed"
project_path = "C:/Users/lazni/PycharmProjects/Age_Predictor"

start = time.perf_counter()

extract_faces(scale_factor=1.01,
              min_neighbours=70,
              img_size=200,
              path_in=in_raw_imgs,
              path_out=out_img,
              data_augmentation=False)
remove_duplicates(path=out_img)

extract_faces(scale_factor=1.3,
              min_neighbours=8,
              img_size=200,
              path_in=in_raw_imgs,
              path_out=out_img,
              data_augmentation=True)
remove_duplicates(path=out_img)
change_brightness(path_in=out_img,
                  prefix1="drk_",
                  factor1=0.9,
                  prefix2="brt_",
                  factor2=1.1)
convert_to_grayscale(path_in=out_img,
                     path_out=out_img)
extract_labels(path_in=out_img,
               path_out_labels=project_path)
convert_to_array(path_in=out_img,
                 path_out=project_path)

end = time.perf_counter()
print(f"Data transformation completed in: {end - start} seconds")
