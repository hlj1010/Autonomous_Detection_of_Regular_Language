import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from random import shuffle
from random import sample

CSV_COLUMN_NAMES = ['dfa', 'input_string']


def get_trial_data(filename,
                   randomize=True,
                   start=None,
                   max_entries_per_dfas=None):
    results = []
    current_dfas = ''
    ctr = 0

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print('{}: {}'.format(row[0], row[1]))
            if current_dfas != row["dfa"]:
                current_dfas = row["dfa"]
                ctr = 0

            if max_entries_per_dfas is not None:
                if ctr >= max_entries_per_dfas:
                    continue

            if start is not None:
                if ctr < start:
                    ctr += 1
                    continue

            results.append(row)
            ctr += 1
            # break

    if randomize is True:
        print('Randomizing data set...')
        results = sample(results, k=len(results))
        print('Randomization complete.')

    return results


def load_data(filename, y_name="input_string", dfa=None):
    data = pd.read_csv(filename, names=CSV_COLUMN_NAMES, header=0)

    if dfa is not None:
        data = data[data['dfa'] == dfa]

    data_x, data_y = data, data.pop(y_name)
    return (data_x, data_y)


def load_train_data(y_name="input_string", dfa=None):
    return load_data("train_data.csv", y_name=y_name, dfa=dfa)
    # train = pd.read_csv("train_data.csv", names=CSV_COLUMN_NAMES, header=0)
    # train_x, train_y = train, train.pop(y_name)
    # return (train_x, train_y)


def load_train_data2(base_filename="C:\\Users\\joels\\Documents\\GitHub\\Autonomous_Detection_of_Regular_Language\\DFAS\\redo\\dfa", qty=100):
    import struct

    data_x = []
    data_y = []

    for i in range(7):
        with open('{}_{}.bin'.format(base_filename, i), 'rb') as fd:
            for _ in range(qty):
                b = fd.read(4)

                if len(b) == 4:
                    data_x.append('dfa_{}'.format(i))
                    data_y.append(struct.unpack('L', b)[0])
                else:
                    break

            fd.close()

    return (data_x, data_y)


def load_test_data2(base_filename="C:\\Users\\joels\\Documents\\GitHub\\Autonomous_Detection_of_Regular_Language\\DFAS\\redo\\dfa", qty=100, skip_first=100):
    import struct

    data_x = []
    data_y = []

    for i in range(7):
        with open('{}_{}.bin'.format(base_filename, i), 'rb') as fd:
            fd.seek(4 * skip_first, 0)
            for _ in range(qty):
                b = fd.read(4)

                if len(b) == 4:
                    data_x.append('dfa_{}'.format(i))
                    data_y.append(struct.unpack('L', b)[0])
                else:
                    break

            fd.close()

    return (data_x, data_y)


def load_test_data(y_name="input_string", dfa=None):
    return load_data("test_data.csv", y_name=y_name, dfa=dfa)

    # test = pd.read_csv("test_data.csv", names=CSV_COLUMN_NAMES, header=0)
    # test_x, test_y = test, test.pop(y_name)
    # return (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def chunk_data2(data):
    r = '{:032b}'.format(data)
    r = [[r[0]], [r[1]], [r[2]], [r[3]], [r[4]], [r[5]], [r[6]], [r[7]],
         [r[8]], [r[9]], [r[10]], [r[11]], [r[12]], [r[13]], [r[14]], [r[15]],
         [r[16]], [r[17]], [r[18]], [r[19]], [r[20]], [r[21]], [r[22]], [r[23]],
         [r[24]], [r[25]], [r[26]], [r[27]], [r[28]], [r[29]], [r[30]], [r[31]]]
    return r


def chunk_data(data):
    retval = []
    ctr = 0
    for r in data:
        # print(type(r))
        retval.append([r[i:i + 1] for i in range(0, len(r))])
        ctr += 1

        # if ctr >= 2:
        #     break
        # exit()
    return np.asarray(retval)


def max_length(data):
    m = 0
    for r in data:
        m = max(m, len(r))

    return m


if __name__ == "__main__":

    (train_x, train_y) = load_train_data(dfa='dfa_0')
    print(chunk_data(train_y))
    # d = get_trial_data("../DFAS/trial_data.csv",
    #                    max_entries_per_dfas=1500,
    #                    randomize=False)
    # with open("train_data.csv", "w") as out:
    #     out.write('dfa,input_string\n')
    #     for v in d:
    #         out.write('{},{}\n'.format(v["dfa"], v["input_string"]))

    # # Record test data
    # d = get_trial_data("../DFAS/trial_data.csv", start=1500, randomize=False)
    # with open("test_data.csv", "w") as out:
    #     out.write('dfa,input_string\n')
    #     for v in d:
    #         out.write('{},{}\n'.format(v["dfa"], v["input_string"]))
