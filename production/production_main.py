import os
import constants as c
import numpy as np
import tensorflow as tf
from production.gui import GUI


# todo: make sure we do not get to the end of a list with arrows
gui = GUI()
gui.extract_faces(path_in=os.path.join(c.PRODUCTION_DIR, "raw_images"),
                  path_out=os.path.join(c.PRODUCTION_DIR, "200_200_color"),
                  prefix="",
                  img_size=c.IMG_SIZE,
                  scale_factor=1.3,
                  min_neighbours=3)
gui.remove_duplicates(path=os.path.join(c.PRODUCTION_DIR, "200_200_color"))
gui.convert_to_grayscale(path_in=os.path.join(c.PRODUCTION_DIR, "200_200_color"),
                         path_out=os.path.join(c.PRODUCTION_DIR, "200_200_gray"))
gui.extract_labels(os.path.join(c.PRODUCTION_DIR, "200_200_gray"))
gui.convert_to_array(os.path.join(c.PRODUCTION_DIR, "200_200_gray"))
gui.predictions(path_in=os.path.join(c.PRODUCTION_DIR, "200_200_gray"),
                image_size=c.IMG_SIZE)
print(gui.model.evaluate(np.array(gui.picture_array)/255,
                         tf.keras.utils.to_categorical(gui.labels_prepared, 6)))
gui.visual_main()
gui.mainloop()