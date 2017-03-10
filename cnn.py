from tensorflow.contrib.learn.python.learn.datasets import base
# noinspection PyUnresolvedReferences
from six.moves import xrange  # pylint: disable=redefined-builtin
import os
import tensorflow as tf
import numpy
import glob
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

# general global variables
CROPPED_IMAGE_DIM = 28
COLOR_CHANNELS = 3
LABELS_FILENAME = "labels.csv"

# training global variables
TRAIN_BATCH_SIZE = 10
TRAIN_EPOCH_SIZE = 10
TRAIN_STEPS = 1
TRAIN_DATA_DIR = "stare_data/train/"

# test global variables
TEST_BATCH_SIZE = 5
TEST_EPOCH_SIZE = 5
TEST_STEPS = 1
TEST_DATA_DIR = "stare_data/test/"


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


def read_image_and_label_list(list_filename):
    f = open(list_filename, 'r')

    image_list = []
    label_list = []

    for line in f:
        filename, filelabel, label_comment = line[:-1].split(',')
        image_list.append(filename)
        label_list.append(int(filelabel))

    images_tensor = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels_tensor = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

    return images_tensor, labels_tensor


def read_gif_from_disk(local_input_queue, train_dir):
    local_label = local_input_queue[1]
    image_filename = train_dir + local_input_queue[0] + ".gif"

    file_contents = tf.read_file(image_filename)
    local_image = tf.image.decode_gif(file_contents)

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(local_image[0], CROPPED_IMAGE_DIM, CROPPED_IMAGE_DIM)
    resized_image = tf.reshape(resized_image, [CROPPED_IMAGE_DIM, CROPPED_IMAGE_DIM, COLOR_CHANNELS])

    # cast resized_image int32 to float32
    reshaped_image = tf.cast(resized_image, tf.float32)

    return reshaped_image, local_label


def read_next_batch(local_input_queue, data_dir, batch_size):
    image, label = read_gif_from_disk(local_input_queue, data_dir)
    return tf.train.batch([image, label], batch_size=batch_size)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# Get the training data
train_images, train_labels = read_image_and_label_list(TRAIN_DATA_DIR + LABELS_FILENAME)
train_input_queue = tf.train.slice_input_producer([train_images, train_labels])
train_image_batch, train_label_batch = read_next_batch(train_input_queue, TRAIN_DATA_DIR, TRAIN_BATCH_SIZE)

# Get the test data
test_images, test_labels = read_image_and_label_list(TEST_DATA_DIR + LABELS_FILENAME)
test_input_queue = tf.train.slice_input_producer([test_images, test_labels])
test_image_batch, test_label_batch = read_next_batch(test_input_queue, TEST_DATA_DIR, TEST_BATCH_SIZE)

# Build TensorFlow model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# training
y_ = tf.placeholder(tf.float32, [None, 10])

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# evaluate the trained model
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("List of used training images:")
    print(sess.run([train_images, train_labels]))
    print("List of used test images:")
    print(sess.run([test_images, test_labels]))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("Reading training images:")
    for step in xrange(TRAIN_STEPS):
        print("Next train image in step ",
              step, ", train batch size ", TRAIN_BATCH_SIZE, ", train epoch size ", TRAIN_EPOCH_SIZE)
        print(sess.run([train_image_batch, train_label_batch]))

    print("Reading test images:")
    for step in xrange(TEST_STEPS):
        print("Next test image in step ", step,
              ", test batch size ", TEST_BATCH_SIZE, ", test epoch size ", TEST_EPOCH_SIZE)
        print(sess.run([test_image_batch, test_label_batch]))

    coord.request_stop()
    coord.join(threads)

    train_labels_one_hot = dense_to_one_hot(train_label_batch, 10)
    test_labels_one_hot = dense_to_one_hot(train_label_batch, 10)

    for step in xrange(TRAIN_STEPS):
        print("Training step", step, "with all data")
        train_step.run(feed_dict={x: train_image_batch, y_: train_labels_one_hot, keep_prob: 0.5})
        train_accuracy = accuracy.eval(feed_dict={x: train_image_batch, y_: train_labels_one_hot, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (step, train_accuracy))

    print("test accuracy %g" % accuracy.eval(feed_dict={x: test_image_batch, y_: test_labels_one_hot, keep_prob: 1.0}))
