import tensorflow as tf
import os
import time
import constants as c

from data_preparation.data_transformation import DataTransformation
from dl_models.model import ModelTraining


# Download data
def main(download=False,
         transform=True,
         train=True):
    if transform:
        # Transform data
        start = time.perf_counter()
        transformer = DataTransformation()

        for folder in os.listdir(c.IMAGES_RAW_DIR):
            current_input = os.path.join(c.IMAGES_RAW_DIR, folder)
            current_output = os.path.join(c.IMAGES_TRANSFORMED_DIR, folder)
            #
            # transformer.extract_faces(scale_factor=1.01,
            #                           min_neighbours=70,
            #                           prefix="o_",
            #                           img_size=c.IMG_SIZE,
            #                           path_in=current_input,
            #                           path_out=current_output)
            # transformer.remove_duplicates(current_output)
            #
            # transformer.extract_faces(scale_factor=1.3,
            #                           prefix="13_",
            #                           min_neighbours=8,
            #                           img_size=c.IMG_SIZE,
            #                           path_in=current_input,
            #                           path_out=current_output)
            # transformer.remove_duplicates(current_output)
            #
            # transformer.change_brightness(path=current_output,
            #                               prefix1="drk_",
            #                               factor1=0.9,
            #                               prefix2="brt_",
            #                               factor2=1.1)
            # transformer.convert_to_grayscale(path_in=current_output,
            #                                  path_out=current_output)
            # transformer.balance_classes(path_in=current_output,
            #                             path_out=os.path.join(c.IMAGES_ADDITIONAL_DIR, folder),
            #                             limit=10_000)
            transformer.extract_labels(current_output)
            transformer.convert_to_array(current_output)
            print(f"{folder} - done", "-" * 60, sep="\n")

        transformer.save_labels(c.PROJECT_PARENT_DIR)
        transformer.save_array(c.PROJECT_PARENT_DIR)

        end = time.perf_counter()
        print(f"Data transformation completed in: \n{(end - start) / 60:.2f} minutes")

    if train:
        # Train model
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        model1 = ModelTraining()
        model1.load_data(path_features="C:/Users/lazni/PycharmProjects/Age_Predictor/picture_array.npy",
                         path_labels="C:/Users/lazni/PycharmProjects/Age_Predictor/labels_prepared.csv")
        model1.change_dtypes()
        model1.one_hot_encode_labels()
        model1.shuffle_data(random_state=100)
        model1.scale_features()
        model1.visualise_shapes()
        model1.split_to_features_and_labels(percentage=0.70)
        model1.split_to_validation_sets(percentage=0.70)
        model1.create_model(epochs=400,
                            batch_size=64,
                            neurons_per_layer=(370, 170),  # (4200, 3000, 2000)
                            dropout=(0.31, 0.31, 0.4),  # (0.31, 0.31, 0.4)
                            activation=("relu", "relu"),
                            input_shape=c.IMG_SIZE)
        model1.compile_model(initial_learning_rate=0.001,
                             final_learning_rate=0.00001)
        model1.fit_model(include_validation=True,
                         monitor="loss",
                         patience=5)
        model1.save_model("trained_model.H5")
        model1.visualise_models_learning()
        model1.evaluate_model(model1.X_test, model1.y_test)
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


if __name__ == '__main__':
    main(download=False,
         transform=False,
         train=True)
