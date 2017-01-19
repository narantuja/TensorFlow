#
# source: https://stackoverflow.com/questions/33648322/tensorflow-image-reading-display
#

import tensorflow as tf
from PIL import Image
import numpy as np

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1

#  list of files to read
filename_queue = tf.train.string_input_producer(['../fundusimages/image001.png', '../fundusimages/image002.png'])


def read_png(filename_queue):
    """Reads and parses examples from PNG data files.
    """

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_png(value)  # use png or jpg decoder based on your files.
    height = 32
    width = 32

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(image, width, height)
    resized_image = tf.reshape(resized_image, [height, width, 3])
    # resized_image = tf.random_crop(image, [height, width, 3])
    print(tf.shape(resized_image))

    return resized_image
    # return _generate_image_batch(resized_image, 1)


def _generate_image_batch(image, batch_size):
    images = tf.train.batch([image], batch_size=batch_size)

    return images

for i in range(2): #length of filename list
    read_png(filename_queue)

