import os
import math
import numpy as np 
import tensorflow as tf 

from tensorflow.python.framework import ops
from utils import *

# flags = tf.app.flags
# flags.DEFINE_integer("epoch", 1000, "Each to train [1000]")
# flags.DEFINE_float("learning_rate", 1.0, "Learning rate [1.0]")

def conv2d(input_, output_, k_h, k_w,
            stddev = 0.02, name = "conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1],output_],
            initializer = tf.truncated_normal_initializer(stddev = stddev))
        conv = tf.nn.conv2d(input_, w, strides =[1, 1, 1, 1], padding = 'VALID')
        return conv 

def highway(input_, size, layer_size = 1, bias = -2, f = tf.nn.relu):
    # t = sigmold(Wy + b)
    # z = t * g(Wy + b) + (1 - t ) * y
    # reference, not sure, need prove !
    output = input_
    for idx in xrange(layer_size):
        output = f(tf.nn.rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))

        transform_gate = tf.sigmoid(
            tf.nn.rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
        carry_gate = 1. - transform_gate

        output = transform_gate * output + carry_gate * input_

        return output