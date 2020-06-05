import multiprocessing
import re

import keras
from keras import Input
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Add
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import Flatten
from keras import Model
from keras.optimizers import SGD
import sys

# plot packages
from matplotlib import pyplot as plt
import seaborn

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import os

import pandas as pd
import numpy as np

from keras_preprocessing import image as KImage


class Dataset(object):
    # Base Dataset class for create train and val dataset

    def __init__(self, config):
        self.config = config
        self._image_shape = tuple(self.config.IMAGE_INPUT_SHAPE)
        self._image_ids = []
        self.image_info = []
        self.class_info = []
        self.source_class_ids = {}
        self.num_classes = 0
        self.class_ids = []
        self.class_names = []
        self.num_images = 0
        self.class_from_source_map = {}
        self.image_from_source_map = {}
        self._image_to_class = {}

    def add_class(self, source, class_name, aug=False):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["name"] == class_name:
                # class_name already added
                return
        self.class_from_source_map[class_name] = len(self.class_info)

        # Add the class
        self.class_info.append({
            "source": source,
            "name": class_name,
            'aug': aug
        })

    def add_image(self, source, image_id, class_name, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path
        }
        image_info.update(kwargs)
        self.add_image_class_map(len(self.image_info), class_name)
        self.image_info.append(image_info)

    def add_image_class_map(self, image_id, class_name):
        if len(self.class_from_source_map) == 0:
            sys.exit('Add class Mapping first')
        if class_name not in self.class_from_source_map:
            sys.exit('unknown class name')
        self._image_to_class[image_id] = self.class_from_source_map[class_name]

    def get_class_for_image(self, image_id):
        if image_id not in self._image_to_class:
            sys.exit('There is no mapping for image id')
        return self._image_to_class[image_id]

    def prepare(self):
        """Prepares the Dataset class for use.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_from_source_map = {clean_name(k): v for k, v in self.class_from_source_map.items()}
        self.class_names = list(self.class_from_source_map.keys())
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

    def get_image(self, image_id):
        path = self.image_info[image_id]['path']
        img = KImage.load_img(path, target_size=self._image_shape[:2])
        return np.array(img)

    @property
    def image_ids(self):
        return self._image_ids

    @property
    def image_shape(self):
        return self._image_shape


class BatchNorm(BatchNormalization):
    def call(self, inputs, training=False):
        return super(BatchNorm, self).call(inputs, training=training)


class Resnet:
    start_epoch = 0

    ckpt_file = None

    log_dir = None

    def __init__(self, mode, config, base_dir, arch='resnet50'):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.arch = arch
        self.config = config
        self.base_dir = base_dir
        self.set_logdir()
        self.model = self.build()

    def build(self):

        # image Input
        imageInput = Input(shape=tuple(self.config.IMAGE_INPUT_SHAPE), name='resnet_img_input')

        # Stage1
        x = ZeroPadding2D((3, 3))(imageInput)
        x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=True, name='conv1')(x)
        x = BatchNorm(name='bn_conv1')(x, training=self.config.TRAIN_BN)
        x = Activation(activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # stage 2
        x = self.conv_block(x, [64, 64, 256], 2, 'a', strides=(1, 1), kernal_size=(3, 3))
        x = self.identity_block(x, [64, 64, 256], 2, 'b', kernal_size=(3, 3))
        x = self.identity_block(x, [64, 64, 256], 2, 'c', kernal_size=(3, 3))

        # stage 3
        x = self.conv_block(x, [128, 128, 512], 3, 'a', kernal_size=(3, 3))
        x = self.identity_block(x, [128, 128, 512], 3, 'b', kernal_size=(3, 3))
        x = self.identity_block(x, [128, 128, 512], 3, 'c', kernal_size=(3, 3))
        x = self.identity_block(x, [128, 128, 512], 3, 'd', kernal_size=(3, 3))

        # stage 4
        x = self.conv_block(x, [256, 256, 1024], 4, block='a', kernal_size=(3, 3))
        block_count = {'resnet50': 5, 'resnet101': 22}[self.arch]
        for i in range(block_count):
            x = self.identity_block(x, [256, 256, 1024], stage=4, block=chr(98 + i), kernal_size=(3, 3))

        # stage 5
        x = self.conv_block(x, [512, 512, 2048], 5, 'a', kernal_size=(3, 3))
        x = self.identity_block(x, [512, 512, 2048], 5, 'b', kernal_size=(3, 3))
        x = self.identity_block(x, [512, 512, 2048], 5, 'c', kernal_size=(3, 3))

        # top layer
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        if self.config.NUMBER_OF_CLASSES == 0:
            sys.exit('Number of class not set. check config')
        output = Dense(self.config.NUMBER_OF_CLASSES, activation='softmax', name='fc')(x)

        model = Model(imageInput, output)

        sgd = SGD(lr=self.config.LEARNING_RATE, decay=self.config.WEIGHT_DECAY, momentum=self.config.MOMENTUM)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        if self.mode == 'training':
            if self.ckpt_file is not None and os.path.isfile(self.ckpt_file):
                model.load_weights(self.ckpt_file)
            return model

        else:
            if self.ckpt_file is not None:
                model.load_weights(self.ckpt_file)
                print("model load with latest weights {}".format(self.ckpt_file))
            return model

    def identity_block(self, input_data, filters, stage, block, use_bais=True, kernal_size=(3, 3),
                       strides=(1, 1)):
        filter1, filter2, filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filter1, (1, 1), strides=strides, use_bias=use_bais,
                   name=conv_name_base + '2a')(input_data)
        x = BatchNorm(name=bn_name_base + '2a')(x, training=self.config.TRAIN_BN)
        x = Activation(activation='relu')(x)

        x = Conv2D(filter2, kernal_size, use_bias=use_bais,
                   name=conv_name_base + '2b', padding='same')(x)
        x = BatchNorm(name=bn_name_base + '2b')(x, training=self.config.TRAIN_BN)
        x = Activation(activation='relu')(x)

        x = Conv2D(filter3, (1, 1), strides=strides, use_bias=use_bais,
                   name=conv_name_base + '2c')(x)
        x = BatchNorm(name=bn_name_base + '2c')(x, training=self.config.TRAIN_BN)

        x = Add()([x, input_data])
        x = Activation(activation='relu', name='res' + str(stage) + str(block) + '_out')(x)

        return x

    def conv_block(self, input_data, filters, stage, block, use_bais=True, kernal_size=(3, 3),
                   strides=(2, 2)):
        filter1, filter2, filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filter1, (1, 1), strides=strides, use_bias=use_bais, name=conv_name_base + '2a')(
            input_data)
        x = BatchNorm(name=bn_name_base + '2a')(x, training=self.config.TRAIN_BN)
        x = Activation(activation='relu')(x)

        x = Conv2D(filter2, kernal_size, use_bias=use_bais,
                   name=conv_name_base + '2b', padding='same')(x)
        x = BatchNorm(name=bn_name_base + '2b')(x, training=self.config.TRAIN_BN)
        x = Activation(activation='relu')(x)

        x = Conv2D(filter3, (1, 1), use_bias=use_bais, name=conv_name_base + '2c')(x)
        x = BatchNorm(name=bn_name_base + '2c')(x, training=self.config.TRAIN_BN)

        shortcut = Conv2D(filter3, (1, 1), strides=strides, use_bias=use_bais,
                          name=conv_name_base + '1')(
            input_data)
        shortcut = BatchNorm(name=bn_name_base + '1')(shortcut,
                                                      training=self.config.TRAIN_BN)

        x = Add()([x, shortcut])
        x = Activation(activation='relu', name='res' + str(stage) + str(block) + '_out')(x)
        return x

    def set_logdir(self):
        """Sets the model log directory and epoch counter.
                """
        self.log_dir = os.path.join(self.base_dir, self.config.LOG_DIR_NAME)
        self.ckpt_file = None

        for file in os.listdir(self.log_dir):
            if file.endswith(".ckpt"):
                epoch_num = int(file.split('.')[0].split('-')[-1])
                if epoch_num > self.start_epoch:
                    self.start_epoch = epoch_num
                    self.ckpt_file = os.path.join(self.log_dir, file)

    def train(self, trainDataset, valDataset, layer='all', augmentation=None, epoch=None):
        if epoch is None:
            epoch = self.config.NUMBER_OF_EPOCH
        assert layer in ['all', 'last']
        self.set_trainable(layer)

        train_data_gen = DataGenerator(trainDataset, self.config, augumentation=augmentation)

        val_data_gen = DataGenerator(valDataset, self.config)

        ckpt_file_path = os.path.join(self.log_dir, self.config.CHECKPOINT_FILE_FORMAT)
        callbacks = [
            TensorBoard(log_dir=self.log_dir,
                        histogram_freq=self.config.TENSOR_BOARD_HISTOGRAM_FREQ,
                        write_graph=self.config.TENSOR_BOARD_WRITE_GRAPH,
                        write_images=self.config.TENSOR_BOARD_WRITE_IMAGES),

            ModelCheckpoint(ckpt_file_path,
                            verbose=self.config.MODEL_CHECKPOINT_VERBOSE,
                            save_weights_only=self.config.MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY)
        ]
        if self.config.EARLY_STOPPING_ENABLE:
            callbacks.append(EarlyStopping(verbose=self.config.EARLY_STOPPING_VERBOSE,
                                           monitor=self.config.EARLY_STOPPING_MONITOR,
                                           patience=self.config.EARLY_STOPPING_PATIENCE,
                                           restore_best_weights=self.config.EARLY_STOPPING_RESTORE_BEST_WEIGHTS))

        n_workers = multiprocessing.cpu_count()
        result = self.model.fit_generator(
            train_data_gen,
            validation_data=val_data_gen,
            epochs=epoch,
            initial_epoch=self.start_epoch,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=n_workers
        )
        return result

    def set_trainable(self, train_layer):
        layer_regrex = {
            'all': '.*',
            'last': r'\bfc'
        }
        train_layer = layer_regrex[train_layer]

        for layer in self.model.layers:
            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(train_layer, layer.name))
            layer.trainable = trainable

            # Print trainable layer names
            if trainable:
                print("{:20}   ({})".format(layer.name,
                                            layer.__class__.__name__))

    def evaluate(self, data: Dataset, ckpt_file=None):
        if ckpt_file is not None:
            self.model.load_weights(ckpt_file)
            print('model load with weight {}'.format(ckpt_file))
        y_true = [data.get_class_for_image(img_id) for img_id in data.image_ids]
        data_gen = DataGenerator(data, data.config)
        probs = self.model.predict_generator(data_gen)
        probs = probs[:data.num_images, :]
        # for img_id in data.image_ids:
        #     probs[img_id] = self.model.predict(np.expand_dims(data.get_image(img_id), axis=0))[0]
        y_pred = [p.argmax() for p in probs]
        print('Accuracy: %.3f' % accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=data.class_names))
        conf_matx = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(conf_matx, index=data.class_names,
                             columns=data.class_names)
        plt.figure(figsize=(20, 20))
        seaborn.heatmap(df_cm, annot=True)


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset: Dataset, config, shuffle=True, augumentation=None):
        """Initialization"""
        self.dataset = dataset
        self.batch_size = config.BATCH_SIZE
        self.mean_pixels = config.MEAN_PIXEL
        self.shuffle = shuffle
        self.augmentation = augumentation
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.dataset.num_images / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = self.dataset.image_ids
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dataset.image_shape))
        Y = np.zeros((self.batch_size, self.dataset.num_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            class_id = self.dataset.get_class_for_image(ID)
            if self.augmentation and self.dataset.class_info[class_id]['aug']:
                X[i] = self.augmentation(image=self.dataset.get_image(ID))
            else:
                X[i] = self.dataset.get_image(ID)
            Y[i][class_id] = 1
        for depth in range(self.dataset.image_shape[-1]):
            X[:, :, :, depth] = (X[:, :, :, depth] - self.mean_pixels[depth]) / 255
        return X, Y
