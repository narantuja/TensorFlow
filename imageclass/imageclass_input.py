import tensorflow as tf

BATCH_SIZE = 1
CROPPED_IMAGE_DIM = 24
COLOR_CHANNELS = 3
NUM_INPUT_FILES = 4


def read_png(filename_queue_local):
    """Reads and parses examples from PNG data files.
    """

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue_local)

    image = tf.image.decode_png(value)  # use png or jpg decoder based on your files.

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(image, CROPPED_IMAGE_DIM, CROPPED_IMAGE_DIM)
    resized_image = tf.reshape(resized_image, [CROPPED_IMAGE_DIM, CROPPED_IMAGE_DIM, COLOR_CHANNELS])
    # resized_image = tf.random_crop(image, [height, width, COLOR_CHANNELS])

    # cast resized_image int32 to float32
    reshaped_image = tf.cast(resized_image, tf.float32)

    # return resized_image
    return _generate_image_batch(reshaped_image, BATCH_SIZE)


def _generate_image_batch(local_image, batch_size):
    images = tf.train.batch([local_image], batch_size=batch_size)

    return images


def read_csv(filename_queue_local):
    """Reads and parses examples from CSV data files.
    """

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue_local)
    record_defaults = [[0]]
    col1 = tf.decode_csv(value, record_defaults=record_defaults)

    return col1


def read_inputs():
    #  list of files to read
    filename_queue = tf.train.string_input_producer(['../fundusimages/image001.png',
                                                     '../fundusimages/image002.png',
                                                     '../fundusimages/image003.png',
                                                     '../fundusimages/image004.png'])

    label_filename_queue = tf.train.string_input_producer(['../fundusimages/labels.csv'])

    image = read_png(filename_queue)
    image_class = read_csv(label_filename_queue)

    return [image, image_class]


inputs = read_inputs()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # for j in range(NUM_INPUT_FILES):
    print(sess.run(inputs))

    coord.request_stop()
    coord.join(threads)
