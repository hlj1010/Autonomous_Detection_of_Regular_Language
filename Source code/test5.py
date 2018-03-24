import numpy as np
import random
from random import sample
import utils
import time
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Activation
from keras import optimizers
from keras.models import load_model


def shuffle_data(input_data, output_data):
    print("Entering shuffle_data()")
    zipped = np.empty([len(input_data)], dtype=tuple)

    print('len(input_data): {}'.format(len(input_data)))
    print('len(output_data): {}'.format(len(output_data)))
    print('"zipping" data')
    for i in range(len(input_data)):
        zipped[i] = (utils.chunk_data2(input_data[i]), output_data[i])
    print('zip complete')
    np.random.shuffle(zipped)
    print('shuffle complete')

    train_input = np.empty([len(input_data), 32, 1])
    train_output = np.empty([len(input_data), 1])
    for i in range(len(zipped)):
        train_input[i] = np.asarray(zipped[i][0])
        train_output[i] = np.asarray(zipped[i][1])

    print('------------------------------------------------------------------')

    print('len(input_data): {}'.format(len(input_data)))
    print('len(output_data): {}'.format(len(output_data)))

    print("Exiting shuffle_data()\n\n")

    return train_input, train_output


dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

NUM_TRAIN_PER_DFA_BIN = 1000000
NUM_TEST_PER_DFA_BIN = 1000000
DATA_WIDTH = 32
TARGET_DFA = 'dfa_0'
MAX_BATCH_SIZE = 3000
EPOCHS = 15000

RELOAD_MODEL = True

(train_x, train_y) = utils.load_train_data2(qty=NUM_TRAIN_PER_DFA_BIN)
NUM_TRAIN = len(train_x)


(test_x, test_y) = utils.load_test_data2(
    qty=NUM_TEST_PER_DFA_BIN, skip_first=NUM_TRAIN_PER_DFA_BIN)
NUM_TEST = len(test_x)


print('len(test_x): {}'.format(len(test_x)))
print('len(test_y): {}'.format(len(test_y)))

# train_input = []

train_output = []
# print(train_x)
for dfa_name in train_x:
    # print(s['dfa'])
    if dfa_name == TARGET_DFA:
        train_output.append([1])
    else:
        train_output.append([0])

test_output = []
# print(train_x)
for dfa_name in test_x:
    # print(s['dfa'])
    if dfa_name == TARGET_DFA:
        test_output.append([1])
    else:
        test_output.append([0])


train_input, train_output = shuffle_data(train_y, train_output)
train_y = None

test_input, test_output = shuffle_data(test_y, test_output)
test_y = None

# train_input = np.asarray(train_input)
# train_output = np.asarray(train_output)
# test_input = np.asarray(test_input)
# test_output = np.asarray(test_output)

print("Testing and training data loaded")


if RELOAD_MODEL is True:
    model = load_model('./model_data/model.h5')

else:

    model = Sequential()
    # model.add(LSTM(1, input_shape=(32, 1), activation='hard_sigmoid', unit_forget_bias=True))
    model.add(CuDNNLSTM(1, input_shape=(DATA_WIDTH, 1), unit_forget_bias=True))

    # # print(model.layers[0].get_config())

    # model.add(CuDNNLSTM(2, input_shape=(2, 1), unit_forget_bias=True))
    model.add(Activation('tanh'))
    # model.add(Activation('softmax'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


print(len(train_input))
print(len(train_output))


try:
    model.fit(train_input, train_output, batch_size=MAX_BATCH_SIZE, epochs=EPOCHS)
    pass
except KeyboardInterrupt as e:
    pass
finally:
    if RELOAD_MODEL is False:
        model.save('./model_data/model.h5')
    score = model.evaluate(test_input, test_output, batch_size=MAX_BATCH_SIZE)
    print(score)
