# Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf

# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn
# sess = tf.InteractiveSession()

NUM_EXAMPLES = 10000

train_input = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(train_input)
train_input = [map(int, i) for i in train_input]
ti = []
for i in train_input:
    temp_list = []
    for j in i:
        temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    temp_list = ([0] * 21)
    temp_list[count] = 1
    train_output.append(temp_list)

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print(type(train_input))
print(len(train_input))
print(type(train_output))
print(len(train_output))
print('------------')
for i in range(4):
    print(train_input[i])
    print(train_output[i])

exit()

print("test and training data loaded")


# Number of examples, number of input, dimension of each input
data = tf.placeholder(dtype=tf.float32, shape=[None, 20, 1])
target = tf.placeholder(dtype=tf.float32, shape=[None, 21])

# num_units / num_hidden
num_hidden = 24

# state_is_tuple : If True, accepted and returned states are 2-tuples of the
# c_state and m_state. If False, they are concatenated along the column axis.
# This latter behavior will soon be deprecated.
cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)

# create the RNN based on the cell.
# returns the RNN output Tensor and the final state.
val, _ = tf.nn.dynamic_rnn(cell=cell, inputs=data, dtype=tf.float32)
print(val)

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
val = tf.transpose(a=val, perm=[1, 0, 2])
print(val)
print(val.get_shape())

last = tf.gather(params=val, indices=(int(val.get_shape()[0]) - 1))
print(last)
print(val.get_shape())
exit()


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
# sess = tf.Session()
sess.run(fetches=init_op)

batch_size = 1000
no_of_batches = int(len(train_input) // batch_size)
epoch = 5000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr +
                               batch_size], train_output[ptr:ptr + batch_size]

        print('inp: {}'.format(inp))
        print('out: {}'.format(out))

        ptr += batch_size

        exit()

        sess.run(fetches=minimize, feed_dict={data: inp, target: out})
    print("Epoch ", str(i))
incorrect = sess.run(error, feed_dict={data: test_input, target: test_output})
print(sess.run(prediction, {data: [[[1], [0], [0], [1], [1], [0], [1], [
      1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1], [0]]]}))
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
