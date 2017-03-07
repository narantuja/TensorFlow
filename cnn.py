from tensorflow.contrib.learn.python.learn.datasets import base
# noinspection PyUnresolvedReferences
from six.moves import xrange  # pylint: disable=redefined-builtin
import os
import tensorflow as tf
import numpy
import glob

CROPPED_IMAGE_DIM = 28
COLOR_CHANNELS = 3


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def read_images(gif_files):
    print('Extracting training images from GIF files.')

    filename_queue = tf.train.string_input_producer(gif_files)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_gif(value)

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(image[0], CROPPED_IMAGE_DIM, CROPPED_IMAGE_DIM)
    resized_image = tf.reshape(resized_image, [CROPPED_IMAGE_DIM, CROPPED_IMAGE_DIM, COLOR_CHANNELS])

    # cast resized_image int32 to float32
    reshaped_image = tf.cast(resized_image, tf.float32)

    return resized_image


train_images = glob.glob("stare_data/train/*.gif")
gifs = read_images(train_images)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for f in train_images:
        print("Next image:")
        print(sess.run(gifs))

    coord.request_stop()
    coord.join(threads)


def extract_labels(f):
    """Extract the labels into a 1D uint8 numpy array [index].

      Args:
        f: A file object that can be passed into a csv reader.

      Returns:
        labels: a 1D unit8 numpy array.
      """
    print('Extracting', f.name)


def read_data_sets(data_dir, validation_size=0):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_images = read_images(train_dir)

    local_file = os.path.join(train_dir, "trainLabels.csv")

    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f)

    test_images = read_images(test_dir)

    local_file = os.path.join(test_dir, "testLabels.csv")

    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


class DataSet(object):
    def __init__(self, images, labels):
        """Construct a DataSet."""
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

#
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# y = tf.nn.softmax(tf.matmul(x, W) + b)
#
# # training
# y_ = tf.placeholder(tf.float32, [None, 10])
#
# # first convolutional layer
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
#
# x_image = tf.reshape(x, [-1, 28, 28, 1])
#
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# # second convolutional layer
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# # densely connected layer
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# # dropout
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# # readout layer
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
# # evaluate the trained model
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # read data
# input_data_sets = read_data_sets("stare_data")
#
# # launch the session
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
#
# for i in range(200):
#     batch = input_data_sets.next_batch(50)
#     if i % 10 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
#         print("step %d, training accuracy %g" % (i, train_accuracy))
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
# print("test accuracy %g" % accuracy.eval(feed_dict={x: input_data_sets.test.images,
#                                                     y_: input_data_sets.test.labels,
#                                                     keep_prob: 1.0}))
