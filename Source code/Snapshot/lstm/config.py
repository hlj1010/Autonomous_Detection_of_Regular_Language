import configparser
import os
import json

###############################################################################
# Workaround to enable CNTK GPU functionality
###############################################################################

# Obtain Keras base dir path: either ~/.keras or /tmp.
_keras_base_dir = os.path.expanduser('~')
if not os.access(_keras_base_dir, os.W_OK):
    _keras_base_dir = '/tmp'
_keras_dir = os.path.join(_keras_base_dir, '.keras')

# Default backend: TensorFlow.
_BACKEND = 'tensorflow'

# Attempt to read Keras config file.
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _backend = _config.get('backend', _BACKEND)
    assert _backend in {'theano', 'tensorflow', 'cntk'}
    _BACKEND = _backend

if _BACKEND == 'cntk':
    # if os.environ['KERAS_BACKEND'] == 'tensorflow':
    import cntk
    from cntk.device import try_set_default_device, gpu
    import cntk as C

    print(C.device.all_devices())
    print(C.device.try_set_default_device(cntk.device.gpu(0)))
    print(C.device.use_default_device())


###############################################################################
# Core configuration file informaiton
###############################################################################
config = configparser.ConfigParser()
config.read('config.ini')

LOG_DATA_PATH = os.path.normpath(config['LOGGING']['LOG_DATA_PATH'])
TENSORBOARD_LOG_DATA = os.path.normpath(
    config['LOGGING']['TENSORBOARD_LOG_DATA'])
CSV_LOG_DATA = os.path.normpath(
    config['LOGGING']['CSV_LOG_DATA'])

DATASET_PATH = os.path.normpath(config['INPUT_DATA']['DATASET_PATH'])
TARGET_DFA = config['INPUT_DATA']['TARGET_DFA']
DATA_WIDTH = config.getint('INPUT_DATA', 'DATA_WIDTH')
MAX_DATA_WIDTH = config.getint('INPUT_DATA', 'MAX_DATA_WIDTH')
NUM_SAMPLES = config.getint('INPUT_DATA', 'NUM_SAMPLES')
MIN_TRAINING_DATA_WIDTH = None
MAX_TRAINING_DATA_WIDTH = None
# NUM_TESTING_SAMPLES = config.getint('INPUT_DATA', 'NUM_TESTING_SAMPLES')
# NUM_TRAINING_SAMPLES = config.getint('INPUT_DATA', 'NUM_TRAINING_SAMPLES')

EPOCHS = config.getint('TRAINING', 'EPOCHS')
ITERATIONS = config.getint('TRAINING', 'ITERATIONS')
K_FOLDS = config.getint('TRAINING', 'K_FOLDS')
MAX_BATCH_SIZE = config.getint('TRAINING', 'MAX_BATCH_SIZE')
CONTIGUOUS_INPUT_DATA = False
RANDOMIZE_INPUT_DATA = False

MODEL_DATA_PATH = os.path.normpath(config['MODEL']['MODEL_DATA_PATH'])
TRAIN_MODEL = config.getboolean('MODEL', 'TRAIN_MODEL')
RELOAD_MODEL = config.getboolean('MODEL', 'RELOAD_MODEL')
