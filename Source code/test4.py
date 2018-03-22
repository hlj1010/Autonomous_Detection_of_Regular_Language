# Derived from using the source code at the blog post at
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/ as an example.

import numpy as np
import random
from random import shuffle
import tensorflow as tf
import utils
import time

import os


def shuffle_data(input_data, output_data):
    train_input = []
    zipped = []

    for i in range(len(input_data)):
        zipped.append((utils.chunk_data2(input_data[i]), output_data[i]))

    shuffle(zipped)

    train_input.clear()
    output_data = []
    for i in range(len(zipped)):
        train_input.append(zipped[i][0])
        train_output.append(zipped[i][1])

    return train_input, train_output


dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn
# sess = tf.InteractiveSession()

NUM_EXAMPLES = 10000

NUM_TRAIN_PER_DFA_BIN = 1000
NUM_TEST_PER_DFA_BIN = 1000

(train_x, train_y) = utils.load_train_data2(qty=NUM_TRAIN_PER_DFA_BIN)
NUM_TRAIN = len(train_x)


(test_x, test_y) = utils.load_test_data2(
    qty=NUM_TEST_PER_DFA_BIN, skip_first=NUM_TRAIN_PER_DFA_BIN)
NUM_TEST = len(test_x)


print('len(test_x): {}'.format(len(test_x)))
print('len(test_y): {}'.format(len(test_y)))

target_dfa = 'dfa_0'
# train_input = []

train_output = []
# print(train_x)
for dfa_name in train_x:
    # print(s['dfa'])
    if dfa_name == target_dfa:
        train_output.append([1])
    else:
        train_output.append([0])

test_output = []
# print(train_x)
for dfa_name in test_x:
    # print(s['dfa'])
    if dfa_name == target_dfa:
        test_output.append([1])
    else:
        test_output.append([0])


data_width = 32

print("Testing and training data loaded")


# Data's dimensions are Unknown rows, with an unknown length, with one
# transition out from the current state.
data = tf.placeholder(dtype=tf.float32, shape=[None, 32, 1])

print('Data has been chunked.')

# The resulting target has a completely unknown shape at this time and is
# not specified.
target = tf.placeholder(dtype=tf.float32, shape=[None, 1])


# num_units / num_hidden
num_hidden = 33

# state_is_tuple : If True, accepted and returned states are 2-tuples of the
# c_state and m_state. If False, they are concatenated along the column axis.
# This latter behavior will soon be deprecated.
# Based on the work of Gres and Schmidhuber, we will use peepholes.
# Also, based on Gres and Schmidhuber's work, we know that the forget_bias will
# need to be evaluated. Currently defaulting to 1.0.
cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden,
                               state_is_tuple=True,
                               use_peepholes=True,
                               forget_bias=1.0)

# create the RNN based on the cell.
# returns the RNN output Tensor and the final state.
# 'output' is a tensor of shape [batch_size, max_time, 256]
# 'state' is a N-tuple where N is the number of LSTMCells containing a
# tf.contrib.rnn.LSTMStateTuple for each cell
output, state = tf.nn.dynamic_rnn(cell=cell, inputs=data, dtype=tf.float32)

# From Tensorflow’s docstring here
# (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py),

# The outputs of dynamic_rnn called before the transpose are,
# If time_major == False (default), this will be a Tensor shaped: [batch_size,
# max_time, cell.output_size]. If time_major == True, this will be a Tensor
# shaped: [max_time, batch_size, cell.output_size].

# We didn’t pass a value for time_major while calling the function so it
# defaults to false in which case the returned tensor dimensions are
# [batch_size, max_time, cell.output_size].

# Tf.transpose transposes the tensor to change to [max_time, batch_size,
# cell.output_size] ([0 1 2] —> [1 0 2]). For our experiment, we need the
# last time step of each batch. The next function call, tf.gather simply gets
# us the last element in the first dimension, which will be the last time step
# for all the batches. Say, if A is transposed 3 dim array, [L X M X N], with
# gather we get A[L]

# Transposes a. Permutes the dimensions according to perm.
print(output.get_shape())
output = tf.transpose(a=output, perm=[1, 0, 2])
print(output.get_shape())

###################################################################
# SKIPPING FOR NOW.  NEED TO COME BACK TO THIS STEP IN ORDER TO
# UNDERSTAND WHAT IT DOES AND IF IT'S NEEDED
last = tf.gather(params=output, indices=(int(output.get_shape()[0]) - 1))
print(output.get_shape())
# print(last)
# exit()

weight = tf.Variable(tf.truncated_normal(
    [num_hidden, int(target.get_shape()[1])]))

bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

cross_entropy = - \
    tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


init_op = tf.global_variables_initializer()

# Configure GPU settings
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    config = config
    save_path = saver.save(sess, "./model_data/model.ckpt")
    print("Model saved in path: %s" % save_path)

    # sess.run(fetches=init_op)

    batch_size = (len(train_y)) // 1
    no_of_batches = int(len(train_y) // batch_size)
    epoch = 500

    print('Batch Size: {}'.format(batch_size))
    print('# of Batches: {}'.format(no_of_batches))
    print('Epoch: {}'.format(epoch))
    print('========================================')

    # preload all training data into memory in order to speed things up


    train_input, train_output = shuffle_data(train_y, train_output)
    test_input, train_output = shuffle_data(test_y, test_output)

    # train_input = []
    # zipped = []

    # for i in range(len(train_y)):
    #     zipped.append((utils.chunk_data2(train_y[i]), train_output[i]))

    # shuffle(zipped)

    # train_input = []
    # train_output = []
    # for i in range(len(zipped)):
    #     train_input.append(zipped[i][0])
    #     train_output.append(zipped[i][1])












    # dataset = tf.data.Dataset.zip((train_input, train_output))
    # dataset = dataset.batch(32)
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()

    # for _ in range(100):
    #     sess.run(iterator.initializer)
    #     while True:
    #         try:
    #             start_time = time.time()
    #             sess.run(next_element)
    #         except tf.errors.OutOfRangeError:
    #             break

    #         print("Epoch {}: Elapsed Time: {}".format(i, time.time() - start_time))

    # session.run(next_element)
    # inp, out = tf.train.shuffle_batch(
    #       [train_input, train_output],
    #       batch_size=32,
    #       num_threads=4,
    #       capacity=50000,
    #       min_after_dequeue=10000)

    # overall_count = 0
    start_time = time.time()
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):

            # for k in range(ptr, ptr + batch_size):

            #     # print('train_y[k]: {}'.format(train_y[k]))
            #     # print('type(train_y[k]): {}'.format(type(train_y[k])))
            #     train_input.append(utils.chunk_data2(train_y[k]))
            #     # print(train_input[0])

            inp = train_input[ptr:ptr + batch_size]
            out = train_output[ptr:ptr + batch_size]

            # print('inp: {}'.format(inp))
            # print('out: {}'.format(out))

            # exit()
            ptr += batch_size

            sess.run(fetches=minimize, feed_dict={data: inp, target: out})
            # sess.run(fetches=minimize, feed_dict={data: inp, target: out})

    # inp = train_input
    # out = train_output

    # for i in range(epoch):
    #     sess.run(fetches=minimize, feed_dict={data: inp, target: out})
    #     print("Epoch {}: Elapsed Time: {}".format(i, time.time() - start_time))
    #     start_time = time.time()

        print("Epoch {}: Elapsed Time: {}".format(i, time.time() - start_time))
        # print('cell.input_shape: {}'.format(cell.input_shape))
        # print('cell.trainable_variables: {}'.format(cell.trainable_variables))
        # print('cell.trainable_weights: {}'.format(cell.trainable_weights))
        # print('cell.weights: {}'.format(cell.weights))
        start_time = time.time()

    test_input = []
    for k in range(0, NUM_TEST):
        test_input.append(utils.chunk_data2(test_y[k]))

    # print('test_input: {}'.format(test_input))
    # print('len(test_input): {}'.format(len(test_input)))
    # print('test_output: {}'.format(test_output))
    # print('len(test_output): {}'.format(len(test_output)))

    incorrect = sess.run(
        error, feed_dict={data: test_input, target: test_output})
    print('type(incorrect): {}'.format(type(incorrect)))
    print('incorrect: {}'.format(incorrect))
    print(sess.run(prediction, {data: test_input[1500:1505]}))
    print(test_output[1500:1505])
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    sess.close()
