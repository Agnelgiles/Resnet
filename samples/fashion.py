import os
import json
import random

import pandas as pd
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT_DIR = os.path.abspath("Resnet/")

sys.path.append(ROOT_DIR)

from resnet.config import Config
from resnet.model import Dataset
from resnet.model import Resnet


class FashionConfig(Config):
    """create Config file with your specific configuration by
     by override Config class"""

    NUMBER_OF_CLASSES = 20

    BATCH_SIZE = 30

    DATA_FRAME_FILE_NAME = 'images.csv'

    LEARNING_RATE = 0.001

    IMAGE_MIN_DIM = 448

    IMAGE_MAX_DIM = 448

    EARLY_STOPPING_ENABLE = False

    WEIGHT_DECAY = 0.00005

    # MEAN_PIXEL = np.array([215.16, 210.46, 208.79])


class FashionDataset(Dataset):
    """
    config:
    """

    def __init__(self, config, image_dir, image_ids, class_names,
                 data_frame, x_column, y_column, aug_class=[]):
        super(FashionDataset, self).__init__(config)
        for class_name in class_names:
            if class_name in aug_class:
                self.add_class('fashion', class_name, aug=True)
            else:
                self.add_class('fashion', class_name)
        for img_id in image_ids:
            img_path = os.path.join(image_dir, str(img_id) + '.jpg')
            class_name = data_frame[data_frame[x_column] == img_id][y_column].iloc[0]
            self.add_image('fashion', img_id, class_name, img_path)

        self.prepare()


def train(image_dir, base_dir, train_data_filename):
    config = FashionConfig()
    train_data_filename = os.path.join(base_dir, train_data_filename)
    with open(train_data_filename) as f:
        train_data = json.load(f)
    config.NUMBER_OF_CLASSES = len(train_data['class_names'])
    dataframe_path = os.path.join(base_dir, config.DATA_FRAME_FILE_NAME)
    image_data = pd.read_csv(dataframe_path)

    x_column = train_data['x_column']
    y_column = train_data['y_column']

    trainDataset = FashionDataset(config, image_dir, train_data['train_ids'], train_data['class_names'], image_data,
                                  x_column,
                                  y_column, aug_class=['Bags', 'Innerwear', 'Lips', 'Loungewear and Nightwear', 'Ties'])

    valDataset = FashionDataset(config, image_dir, train_data['val_ids'], train_data['class_names'], image_data,
                                x_column,
                                y_column)

    resnet = Resnet('training', config, base_dir, arch='resnet50')

    augmentation = iaa.SomeOf(1, [
        iaa.AdditiveGaussianNoise(scale=0.15 * 255),
        iaa.Noop(),
        iaa.MotionBlur(k=18),
        iaa.Fliplr(),
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.CropToFixedSize(width=trainDataset.image_shape[0], height=trainDataset.image_shape[1])
    ], random_order=True)

    result = resnet.train(trainDataset, valDataset, layer='all', epoch=20, augmentation=augmentation)
    return result


def evaluate(image_dir, base_dir, train_data_filename, ckpt_file=None):
    config = FashionConfig()
    train_data_filename = os.path.join(base_dir, train_data_filename)
    with open(train_data_filename) as f:
        train_data = json.load(f)
    dataframe_path = os.path.join(base_dir, config.DATA_FRAME_FILE_NAME)
    image_data = pd.read_csv(dataframe_path)
    x_column = train_data['x_column']
    y_column = train_data['y_column']
    testDataset = FashionDataset(config, image_dir, train_data['test_ids'], train_data['class_names'], image_data,
                                 x_column,
                                 y_column)

    resnet = Resnet('inference', config, base_dir, arch='resnet50')
    if ckpt_file is not None:
        ckpt_file = os.path.join(base_dir, ckpt_file)
    resnet.evaluate(testDataset, ckpt_file)


def get_data(image_dir, base_dir, train_data_filename):
    config = FashionConfig()
    train_data_filename = os.path.join(base_dir, train_data_filename)
    print(train_data_filename)
    with open(train_data_filename) as f:
        train_data = json.load(f)
    x_column = train_data['x_column']
    y_column = train_data['y_column']

    dataframe_path = os.path.join(base_dir, config.DATA_FRAME_FILE_NAME)
    image_data = pd.read_csv(dataframe_path)
    return FashionDataset(config, image_dir, train_data['val_ids'], train_data['class_names'], image_data,
                          x_column,
                          y_column)


def display_random_data(data: FashionDataset):
    augmentation = iaa.SomeOf(1, [
        iaa.AdditiveGaussianNoise(scale=0.15 * 255),
        iaa.Noop(),
        iaa.MotionBlur(k=18),
        iaa.Fliplr(),
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.CropToFixedSize(width=data.image_shape[0], height=data.image_shape[1])
    ], random_order=True)

    selected_image_ids = random.sample(list(data.image_ids), 4)
    selected_images = [data.get_image(im_id) for im_id in selected_image_ids]
    selected_images_class = [data.class_names[data.get_class_for_image(im_id)] for im_id in selected_image_ids]

    plt.figure(figsize=(20, 10))
    plt.imshow(np.hstack(selected_images))
    plt.figtext(.5, .75, 'Orginal picture', fontsize=30, ha='center')
    plt.figtext(.5, .60, selected_images_class, fontsize=20, ha='center')
    plt.figure(figsize=(20, 10))
    plt.figtext(.5, .75, 'augmented picture', fontsize=30, ha='center')
    plt.imshow(np.hstack(augmentation(images=selected_images)))


def get_model(base_dir):
    config = FashionConfig()
    return Resnet('training', config, base_dir, arch='resnet50')
