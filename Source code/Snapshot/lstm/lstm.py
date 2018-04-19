import os
import sys

import numpy as np
import random
import argparse
from random import sample
import time

import utils
import config

import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Permute, Conv1D
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Activation
from keras import optimizers
from keras.models import load_model
from keras import callbacks
from keras import metrics


if __name__ == "__main__":
    args = utils.parse_arguments()

    # print(args)
    # print(args.reload)
    # print('type(args.reload): {}'.format(type(args.reload)))
    # print(args.no_train)
    # print('type(args.no_train): {}'.format(type(args.no_train)))

    if args.reload is not None:
        config.RELOAD_MODEL = args.reload

    if args.no_train is not None:
        config.TRAIN_MODEL = args.no_train

    if args.samples is not None:
        config.NUM_SAMPLES = args.samples

    if args.epochs is not None:
        config.EPOCHS = args.epochs

    if args.iterations is not None:
        config.ITERATIONS = args.iterations

    config.K_FOLDS = args.k_folds

    if args.batch_size is not None:
        config.MAX_BATCH_SIZE = args.batch_size

    if args.dataset_path is not None:
        config.DATASET_PATH = args.dataset_path

    if args.data_width is not None:
        config.DATA_WIDTH = args.data_width

    if args.max_data_width is not None:
        config.MAX_DATA_WIDTH = args.max_data_width

    if args.min_training_data_width is not None:
        config.MIN_TRAINING_DATA_WIDTH = args.min_training_data_width

    if args.max_training_data_width is not None:
        config.MAX_TRAINING_DATA_WIDTH = args.max_training_data_width

    if (((config.MAX_TRAINING_DATA_WIDTH is None) or (config.MIN_TRAINING_DATA_WIDTH is None)) and
            ((config.MAX_TRAINING_DATA_WIDTH is not None) or (config.MIN_TRAINING_DATA_WIDTH is not None))):

        raise("--max-training-data-width or --min-training-data-width is set but the other is not.  Provide a value for both.")

    print('config.MIN_TRAINING_DATA_WIDTH: {}'.format(
        config.MIN_TRAINING_DATA_WIDTH))
    print('config.MAX_TRAINING_DATA_WIDTH: {}'.format(
        config.MAX_TRAINING_DATA_WIDTH))

    if config.MIN_TRAINING_DATA_WIDTH is None:
        config.MIN_TRAINING_DATA_WIDTH = config.DATA_WIDTH
        config.MAX_TRAINING_DATA_WIDTH = config.DATA_WIDTH
    else:
        config.DATA_WIDTH = config.MAX_TRAINING_DATA_WIDTH

    config.RANDOMIZE_INPUT_DATA = args.randomize_input_data
    if config.RANDOMIZE_INPUT_DATA is True:
        config.CONTIGUOUS_INPUT_DATA = False
    else:
        config.CONTIGUOUS_INPUT_DATA = args.continguous_input_data

    print('Starting with the following configuration:')
    print('------------------------------------------')
    print('    Behavioral Parameters:')
    print('\tTrain the Model?     : {}'.format(config.TRAIN_MODEL))
    print('\tReload the Model?    : {}'.format(config.RELOAD_MODEL))

    print('    Input Data Parameters:')
    print('\tDATASET_PATH         : {}'.format(config.DATASET_PATH))
    print('\tDATA_WIDTH           : {}'.format(config.DATA_WIDTH))
    print('\tMAX_DATA_WIDTH       : {}'.format(config.MAX_DATA_WIDTH))
    print('\tTARGET_DFA (Truth)   : {}'.format(config.TARGET_DFA))
    print('\tNUM_SAMPLES (per DFA): {}'.format(config.NUM_SAMPLES))
    print('\tCONTIGUOUS_INPUT_DATA: {}'.format(config.CONTIGUOUS_INPUT_DATA))
    print('\tRANDOMIZE_INPUT_DATA : {}'.format(config.RANDOMIZE_INPUT_DATA))

    print('    Trainig Parameters:')
    print('\tEPOCHS               : {}'.format(config.EPOCHS))
    print('\tITERATIONS           : {}'.format(config.ITERATIONS))
    print('\tK_FOLDS              : {}'.format(config.K_FOLDS))
    print('\tMAX_BATCH_SIZE       : {}'.format(config.MAX_BATCH_SIZE))

    print('    Model Data:')
    print('\tMODEL_DATA_PATH      : {}'.format(config.MODEL_DATA_PATH))

    print('    Logging Data:')
    print('\tLOG_DATA_PATH        : {}'.format(config.LOG_DATA_PATH))
    print('\tLOG_DATA_PATH        : {}'.format(config.LOG_DATA_PATH))
    print('------------------------------------------')
    print()

    if not os.path.exists(config.MODEL_DATA_PATH):
        os.makedirs(config.MODEL_DATA_PATH)

    if not os.path.exists(config.LOG_DATA_PATH):
        os.makedirs(config.LOG_DATA_PATH)

    if not os.path.exists(config.TENSORBOARD_LOG_DATA):
        os.makedirs(config.TENSORBOARD_LOG_DATA)

    ##########################################################
    # Load Data
    ##########################################################
    print("Loading Training and Testing data:")
    print("\tLoading data ... ", end='')
    sys.stdout.flush()
    start_time = time.time()

    input_data = None
    output_data = None

    for dw in range(config.MIN_TRAINING_DATA_WIDTH, config.MAX_TRAINING_DATA_WIDTH+1):
        step_input_data, step_output_data = utils.load_data(
            data_width=dw,
            target_dfa=config.TARGET_DFA,
            base_path=config.DATASET_PATH,
            qty_per_dfa=config.NUM_SAMPLES,
            load_contiguous=config.CONTIGUOUS_INPUT_DATA,
            load_random=config.RANDOMIZE_INPUT_DATA,
            max_data_size=config.MAX_DATA_WIDTH,
        )

        if input_data is None:
            input_data = step_input_data
            output_data = step_output_data
        else:
            input_data = np.append(input_data, step_input_data, axis=0)
            output_data = np.append(output_data, step_output_data, axis=0)

    print("COMPLETED! {:.4f}s".format(time.time() - start_time))
    print("Training and Testing data loaded")

    if (len(input_data) // 2) < config.NUM_SAMPLES:
        print('WARNING: The number of samples loaded from the input files ({}) is less than the numbe requested ({})'.format(
            (len(input_data) // 2), config.NUM_SAMPLES))

    if (len(input_data) < 1):
        raise('No input data was loaded')

    # Hack to allow for optimization when loading different sample sizes from different files.
    # Not sure what the upper limit is for the video card(s).  Guessing it's 200,000 or less.
    config.MAX_BATCH_SIZE = min(50000, len(input_data))

    ##########################################################
    # Prepare Model
    ##########################################################

    if config.RELOAD_MODEL is True:
        ##########################################################
        # Load Model
        ##########################################################
        model = load_model(config.MODEL_DATA_PATH + '/model.h5')

    else:
        ##########################################################
        # Build Model
        ##########################################################

        with tf.device('/cpu:0'):
            model = Sequential()

            if config._BACKEND == 'tensorflow':
                print('INFORMATIONAL: Creating model using CuDNNLSTM')
                model.add(
                    CuDNNLSTM(1,
                              input_shape=(config.MAX_DATA_WIDTH, 1),
                              unit_forget_bias=True
                              )
                )
            else:
                print('INFORMATIONAL: Creating model using LSTM')
                model.add(
                    LSTM(1,
                         input_shape=(config.MAX_DATA_WIDTH, 1),
                         activation='tanh',
                         recurrent_activation='sigmoid',
                         unit_forget_bias=True,
                         recurrent_dropout=0.99,
                         )
                )

            model.add(Activation('sigmoid'))
            # model.add(Activation('tanh'))
            model.compile(loss='binary_crossentropy',
                               optimizer='rmsprop',
                               metrics=[
                                   metrics.binary_accuracy,
                                   metrics.binary_crossentropy,
                                   metrics.mae,
                               ])

        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(loss='binary_crossentropy',
                               optimizer='rmsprop',
                               metrics=[
                                   metrics.binary_accuracy,
                                   metrics.binary_crossentropy,
                                   metrics.mae,
                               ])

    ############################################################
    # k-fold train/test
    ############################################################
    training_start_time = time.time()
    timestamp = str(training_start_time)
    running_score = []
    plateaued = False
    try:
        print('Performing {} iteration(s) of training.'.format(config.ITERATIONS))
        for iteration in range(config.ITERATIONS):
            if plateaued is True:
                break

            print('Starting iteration #{}'.format(iteration))

            for d, v_in, v_out, t_in, t_out in utils.kfold(input_data, output_data, config.K_FOLDS):
                print('Performing k-fold #{:02}'.format(d))

                if config.TRAIN_MODEL is True:

                    if config.RANDOMIZE_INPUT_DATA is True:
                        tf_log_dir = os.path.join(
                            config.TENSORBOARD_LOG_DATA, 'rand')
                    elif config.CONTIGUOUS_INPUT_DATA is True:
                        tf_log_dir = os.path.join(
                            config.TENSORBOARD_LOG_DATA, 'cont')
                    else:
                        tf_log_dir = os.path.join(
                            config.TENSORBOARD_LOG_DATA, 'step')

                    tf_log_dir = os.path.join(
                        tf_log_dir, str(config.NUM_SAMPLES))
                    tf_log_dir = os.path.join(
                        tf_log_dir, str(config.DATA_WIDTH))
                    tf_log_dir = os.path.join(tf_log_dir, str(config.EPOCHS))
                    tf_log_dir = os.path.join(
                        tf_log_dir, str(config.ITERATIONS))
                    tf_log_dir = os.path.join(tf_log_dir, str(iteration))
                    tf_log_dir = os.path.join(tf_log_dir, timestamp)
                    tf_log_dir = os.path.join(tf_log_dir, 'd{:02}'.format(d))
                    os.makedirs(tf_log_dir, exist_ok=True)

                    print("Logging TensorBoard data to: {}".format(tf_log_dir))

                    cb_tb = callbacks.TensorBoard(log_dir=tf_log_dir,
                                                  histogram_freq=0,
                                                  write_graph=True,
                                                  write_grads=False,
                                                  write_images=False)

                    if config.RANDOMIZE_INPUT_DATA is True:
                        csv_log_dir = os.path.join(
                            config.CSV_LOG_DATA, 'rand')
                    elif config.CONTIGUOUS_INPUT_DATA is True:
                        csv_log_dir = os.path.join(
                            config.CSV_LOG_DATA, 'cont')
                    else:
                        csv_log_dir = os.path.join(
                            config.CSV_LOG_DATA, 'step')

                    csv_log_dir = os.path.join(
                        csv_log_dir, str(config.NUM_SAMPLES))
                    csv_log_dir = os.path.join(
                        csv_log_dir, str(config.DATA_WIDTH))
                    csv_log_dir = os.path.join(csv_log_dir, str(config.EPOCHS))
                    csv_log_dir = os.path.join(
                        csv_log_dir, str(config.ITERATIONS))
                    csv_log_dir = os.path.join(csv_log_dir, str(iteration))
                    csv_log_dir = os.path.join(csv_log_dir, timestamp)
                    csv_log_dir = os.path.join(csv_log_dir, 'd{:02}'.format(d))
                    os.makedirs(csv_log_dir, exist_ok=True)

                    print("Logging CSV data to: {}".format(csv_log_dir))

                    cb_csv = callbacks.CSVLogger(
                        os.path.join(csv_log_dir, 'training.csv'))

                    # print(
                    #     "Setting Reduce Learning Rate on Plateau to monitor the 'mean_absolute_error'")
                    # cb_lrop = callbacks.ReduceLROnPlateau(
                    #     monitor='mean_absolute_error',
                    #     factor=0.1,
                    #     patience=5,
                    #     verbose=0,
                    #     mode='min',
                    #     epsilon=0.001,
                    #     cooldown=10,
                    #     min_lr=0
                    # )

                    print(
                        "Setting Early Stopping to monitor the 'mean_absolute_error' for a change of 0.0001 and delta of 1000.")
                    cb_es = callbacks.EarlyStopping(
                        monitor='mean_absolute_error', min_delta=0.0001, patience=5, mode='min')

                    # cb_plateaued = LambdaCallback(
                    #     on_train_end=lambda mean_absolute_error: print(mean_absolute_error))


                    parallel_model.fit(
                        t_in,
                        t_out,
                        batch_size=config.MAX_BATCH_SIZE,
                        epochs=config.EPOCHS,
                        callbacks=[cb_tb, cb_csv, cb_es]
                    )
                pass

                print("Evaluating Performance...")

                score = parallel_model.evaluate(
                    v_in,
                    v_out,
                    batch_size=config.MAX_BATCH_SIZE
                )
                print(score)
                running_score.append(score)

            score = parallel_model.evaluate(
                input_data,
                output_data,
                batch_size=config.MAX_BATCH_SIZE
            )
    except KeyboardInterrupt as e:
        pass
    finally:
        if config.TRAIN_MODEL is True:
            print('Training duration: {:.4f} seconds'.format(
                time.time() - training_start_time))
            print("Saving Model")

            # model_file_path = os.path.join(
            #     config.MODEL_DATA_PATH, 'model.h5')
            # model_json_file_path = os.path.join(
            #     config.MODEL_DATA_PATH, 'model.json')

            # model_json = parallel_model.to_json()
            # with open(model_json_file_path, "w") as json_file:
            #     json_file.write(model_json)
            # # serialize weights to HDF5
            # model.save_weights(model_file_path)

            # model_file_path = os.path.join(config.MODEL_DATA_PATH,
            #                                '{}_model.h5'.format(timestamp))
            # model_json_file_path = os.path.join(config.MODEL_DATA_PATH,
            #                                '{}_model.json'.format(timestamp))
            # model_json = parallel_model.to_json()
            # with open(model_json_file_path, "w") as json_file:
            #     json_file.write(model_json)
            # # serialize weights to HDF5
            # model.save_weights(model_file_path)

            model.save(config.MODEL_DATA_PATH + '/model.h5')
            model.save(os.path.join(config.MODEL_DATA_PATH,
                                    '{}_model.h5'.format(timestamp)))

            print("Saved model to disk")

    t_loss = 0.0
    t_binary_accuracy = 0.0
    t_binary_crossentropy = 0.0
    t_mean_absolute_error = 0.0

    for loss, binary_accuracy, binary_crossentropy, mean_absolute_error in running_score:
        t_loss += loss
        t_binary_accuracy += binary_accuracy
        t_binary_crossentropy += binary_crossentropy
        t_mean_absolute_error += mean_absolute_error

    t_loss = t_loss / len(running_score)
    t_binary_accuracy = t_binary_accuracy / len(running_score)
    t_binary_crossentropy = t_binary_crossentropy / len(running_score)
    t_mean_absolute_error = t_mean_absolute_error / len(running_score)
    print('Average Scores:')
    print('\tAvg. Total Loss: {}'.format(t_loss))
    print('\tAvg. Total Binary Accuracy: {}'.format(t_binary_accuracy))
    print('\tAvg. Total Binary Cross-Entropy: {}'.format(t_binary_crossentropy))
    print('\tAvg. Total Mean Absolute Error: {}'.format(t_mean_absolute_error))
    print('Final Pass:')
    print('\tLoss: {}'.format(score[0]))
    print('\tAccuracy: {}'.format(score[1]))
