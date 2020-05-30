import numpy as np


class Config(object):
    TRAIN_BN = True

    WEIGHT_DECAY = 0.0001

    MOMENTUM = 0.9

    IMAGE_CHANNEL_COUNT = 3

    IMAGE_MIN_DIM = 512

    IMAGE_MAX_DIM = 512

    LEARNING_RATE = 0.001

    NUMBER_OF_EPOCH = 50

    BATCH_SIZE = 35

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # set in your custom config
    NUMBER_OF_CLASSES = 0

    # Tensor board config
    TENSOR_BOARD_HISTOGRAM_FREQ = 0

    TENSOR_BOARD_WRITE_GRAPH = True

    TENSOR_BOARD_WRITE_IMAGES = False

    # ModelCheckpoint config
    MODEL_CHECKPOINT_VERBOSE = 0

    MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY = True

    # EarlyStopping
    EARLY_STOPPING_ENABLE = True

    EARLY_STOPPING_VERBOSE = 1

    EARLY_STOPPING_MONITOR = 'val_loss'

    EARLY_STOPPING_PATIENCE = 3

    EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True

    DATA_FRAME_FILE_NAME = "data.csv"

    LOG_DIR_NAME = 'log'

    CHECKPOINT_FILE_FORMAT = 'resnet101-cp-{epoch:04d}.ckpt'

    VERBOSE = 1

    def __init__(self):
        self.IMAGE_INPUT_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MIN_DIM, self.IMAGE_CHANNEL_COUNT])
        self.WEIGHT_DECAY = self.LEARNING_RATE / self.NUMBER_OF_EPOCH

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
