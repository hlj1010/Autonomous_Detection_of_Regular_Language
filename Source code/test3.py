# Derived from using the source code at the blog post at
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/ as an example.

import numpy as np
import random
from random import shuffle
import tensorflow as tf
import utils

# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn
# sess = tf.InteractiveSession()

NUM_EXAMPLES = 10000

(train_x, train_y) = utils.load_train_data()

target_dfa = 'dfa_0'
# train_input = []
print(type(train_y))
print(train_y)
# for i, v in train_y.iteritems():
#     # print('i: {}'.format(i))
#     # print('v: {}'.format(v))
#     # print(utils.chunk_data(v))
#     train_input.append(utils.chunk_data(v))
# #     exit()

# print('len(train_input): {}'.format(len(train_input)))
# print('len(train_input[0]: {}'.format(len(train_input[0])))
# print('---------------------------------------------------------------')

train_output = []
# print(train_x)
for i, s in train_x.iterrows():
    # print(s['dfa'])
    if s['dfa'] == target_dfa:
        train_output.append(1)
    else:
        train_output.append(0)

(test_x, test_y) = utils.load_test_data()

data_width = max(utils.max_length(train_y), utils.max_length(test_y))
# print('data_width: {}'.format(data_width))
# exit()

# print(train_y[0])
# print(test_x)
print("Testing and training data loaded")


# Data's dimensions are Unknown rows, with an unknown length, with one
# transition out from the current state.
data = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

print('Data has been chunked.')

# The resulting target has a completely unknown shape at this time and is
# not specified.
target = tf.placeholder(dtype=tf.float32, shape=[None, data_width + 1])


# num_units / num_hidden
num_hidden = 24

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
print(output)
print(state)
print('-------------------')
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
output = tf.transpose(a=output, perm=[1, 0, 2])
print(output)
print(output.get_shape())


###################################################################
# SKIPPING FOR NOW.  NEED TO COME BACK TO THIS STEP IN ORDER TO
# UNDERSTAND WHAT IT DOES AND IF IT'S NEEDED
last = tf.gather(params=output, indices=(int(output.get_shape()[0]) - 1))
# print(last)
# print(output.get_shape())
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

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(fetches=init_op)

batch_size = 1
no_of_batches = int(len(train_y) // batch_size)
epoch = 5000

print('Batch Size: {}'.format(batch_size))
print('# of Batches: {}'.format(no_of_batches))
print('Epoch: {}'.format(epoch))


# overall_count = 0
for i in range(epoch):
    ptr = 0

    # print(type(train_y))
    # print(train_y)

    for j in range(no_of_batches):
        # print('len(train_input): {}'.format(len(train_input)))
        # print('len(train_input[j]): {}'.format(len(train_input[j])))
        # print(type(len(train_input[j])))
        # print(train_input[j])
        # print(train_input[j][0])
        # print(train_input[j][1])

        train_input = []
        for k in range(ptr, batch_size):
        # for i, v in train_y.iteritems():
            # print('i: {}'.format(i))
            # print('v: {}'.format(v))
            # print(utils.chunk_data(v))
            print(train_y[k])
            train_input.append(utils.chunk_data(train_y[k]))

        train_input = np.array(train_input)

        inp, out = train_input[ptr:ptr +
                               batch_size], train_output[ptr:ptr + batch_size]

        # print('inp: {}'.format(inp))
        # print('out: {}'.format(out))

        ptr += batch_size

        # overall_count += 1

        # exit()

        sess.run(fetches=minimize, feed_dict={data: inp, target: out})
    print("Epoch ", str(i))

incorrect = sess.run(error, feed_dict={data: test_input, target: test_output})
print(sess.run(prediction, {data: [[[1], [0], [0], [1], [1], [0], [1], [
      1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1], [0]]]}))
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
