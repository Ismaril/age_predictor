import constants as c
import numpy as np
import tensorflow as tf

from production.gui import GUI

gui = GUI()
gui.images_present_raw_dir()
gui.extract_faces(path_in=c.PRODUCTION_RAW_DIR,
                  path_out=c.PRODUCTION_TRANSFORMED_DIR,
                  prefix="",
                  img_size=c.IMG_SIZE,
                  scale_factor=1.01,
                  min_neighbours=70)
gui.remove_duplicates(path=c.PRODUCTION_TRANSFORMED_DIR)
gui.convert_to_grayscale(path_in=c.PRODUCTION_TRANSFORMED_DIR,
                         path_out=c.PRODUCTION_TRANSFORMED_DIR)
gui.images_present_transformed_dir()
gui.extract_labels(c.PRODUCTION_TRANSFORMED_DIR)
gui.convert_to_array(c.PRODUCTION_TRANSFORMED_DIR)
gui.predictions(path_in=c.PRODUCTION_TRANSFORMED_DIR,
                image_size=c.IMG_SIZE)

print(gui.model.evaluate(np.array(gui.picture_array) / 255,
                         tf.keras.utils.to_categorical(gui.labels_prepared, 6)))

gui.visual_main()
gui.mainloop()
