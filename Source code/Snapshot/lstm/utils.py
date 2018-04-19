import os
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from random import shuffle
from random import sample
from random import randrange
import struct
import argparse


def chunk_data32(data, data_width, max_data_size):
    r = ('{:0' + str(data_width) + 'b}').format(data)
    arr = np.zeros(shape=(max_data_size, 1), dtype=np.uint8)
    for i in range(data_width):
        # Remap the DFA symbols.  The resulting array will contain zeros
        # where no DFA symbol is present a 3 where the DFA's 0 symbol was
        # and 6 where the DFA's 1 symbol was.
        arr[i] = (int(r[i]) + 1) * 3
    return arr


def load_data(target_dfa, base_path, data_width, base_filename='dfa', qty_per_dfa=100, num_dfas=2, load_contiguous=False, load_random=False, max_data_size=32):

    f_sizes = []

    for i in range(num_dfas):
        s = os.stat(os.path.join(
            base_path, '{}_{}_{}.bin'.format(data_width, base_filename, i))).st_size
        if s != 0:
            f_sizes.append(s)

    min_entries = min(f_sizes)

    # print('{}_{}_{}.bin: {}'.format(data_width, base_filename, i, f_sizes))

    if qty_per_dfa > (min_entries / 4):       # 4 for an unsigned long
        print("WARNING: The DFA's data does not have enough entries to provide a "
                        "data set of {} samples per DFA!!!  Only {} samples per DFA will be used!".format(qty_per_dfa, qty_per_dfa))
        qty_per_dfa = min_entries // 4

    data = np.empty([qty_per_dfa*num_dfas, 2], dtype=np.uint32)

    j = 0
    for i in range(num_dfas):
        current_dfa = 'dfa_{}'.format(i)
        if target_dfa == current_dfa:
            truth = True
        else:
            truth = False
        with open(os.path.join(base_path, '{}_{}.bin'.format(base_filename, i)), 'rb') as fd:
            s = os.stat(os.path.join(
                base_path, '{}_{}.bin'.format(base_filename, i))).st_size
            step = s % 4
            # print(os.path.join(base_path, '{}_{}.bin'.format(base_filename, i)))

            # filling with ones because an offset of 1 can never be selected whereas a zero
            # could be.
            if load_random is True:
                offset_list = np.ones(qty_per_dfa)

            for k in range(qty_per_dfa):
                if load_contiguous is not False:
                    fd.seek(k * (s // qty_per_dfa), 0)
                elif load_random is True:
                    while True:
                        rnd = randrange(0, s-4)
                        rnd = (rnd % 4) * 4         # Byte align
                        if np.isin(rnd, offset_list[:k]).item(0) is False:
                            break
                        offset_list[k] = rnd

                    fd.seek(rnd, 0)

                b = fd.read(4)

                if len(b) == 4:
                    data[j][0] = truth
                    # print('type(b): {}'.format(type(b)))
                    data[j][1] = struct.unpack('I', b)[0]
                    j += 1
                else:
                    break

            fd.close()
    data = data[:j]
    shuffle(data)

    inputs = np.zeros([len(data), max_data_size, 1], dtype=np.uint8)
    outputs = np.empty(len(data), dtype=np.uint8)
    # , config.MAX_DATA_WIDTH, 1])

    for i in range(len(data)):
        outputs[i] = data[i][0]
        inputs[i] = chunk_data32(data[i][1], data_width, max_data_size)
        # temp[i] = utils.chunk_data32(data[i])

    # # data = temp
    # np.random.shuffle(temp)
    return inputs, outputs

    pass


def kfold(input_data, output_data, k, randomize=True):
    """Given an numpy array and k will return the validation set input/output set and the training input/output set in that order."""
    # np.random.shuffle(input_data)

    # print(input_data.shape)
    # print(output_data.shape)

    if randomize is True:    
        state = np.random.get_state()
        np.random.set_state(state)
        np.random.shuffle(input_data)
        np.random.set_state(state)
        np.random.shuffle(output_data)

        # s = np.dstack((input_data, output_data))
        # np.random.shuffle(s[0])

        # sp = np.dsplit(s, 2)
        # print(sp)
        # print(sp[0][0])
        # input_data = np.transpose(sp[0][0])[0]
        # output_data = np.transpose(sp[1][0])[0]
    
    ld = len(input_data)
    start= 0
    d = 0

    if k == 0:
        yield d, input_data, output_data, input_data, output_data
    else:
        for i in range(max(ld//k, 1), ld+1, max(ld//k, 1)):
            d += 1
            yield d, input_data[start:i], output_data[start:i], np.concatenate([input_data[0:start], input_data[i:]]), np.concatenate([output_data[0:start], output_data[i:]])
            start = i


def parse_arguments():
    parser = argparse.ArgumentParser()
    group1 = parser.add_argument_group('Model/Training Behavior:')
    group1.add_argument(
        '-l', '--reload', action='store_true', help='reload model')
    group1.add_argument('-n', '--no-train',
                        action='store_false', help='train model')
    group1.add_argument('-k', '--k-folds', type=int, default=10, help='number of k-folds')
    group1.add_argument('-e', '--epochs', type=int, help='number of epochs')
    group1.add_argument('-i', '--iterations', type=int, help='number of times to present the same data to the model')
    group1.add_argument('-b', '--batch-size', type=int, help='batch size')
    group1.add_argument('--model-path', type=str, help='path to the model to reload')
    
    group2 = parser.add_argument_group('Training Data:')
    group2.add_argument('-p', '--dataset-path', type=str, help='the path to the data sets')
    group2.add_argument('-d', '--data-width', type=int, default=32, help='the width of the input data')
    group2.add_argument('--min-training-data-width', type=int, help='the minimum width of the input data.  requires --max-training-data-with to be set.')
    group2.add_argument('--max-training-data-width', type=int, help='the maximum width of the input data.  requires --min-training-data-with to be set.')
    group2.add_argument('-m', '--max-data-width', type=int, help='maximum string width accepted by the model')
    group2.add_argument('-r', '--randomize-input-data',
                        action='store_true', help='randomly read in data from the DFAs')
    group2.add_argument('-c', '--continguous-input-data',
                        action='store_true', help='if set, reads input data contiguosly until the request number of samples has been read')
    group2.add_argument('-s', '--samples', type=int,
                        help='the number of samples to be taken per DFA / class')

    return parser.parse_args()
