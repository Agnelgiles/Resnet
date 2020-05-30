import os
import json
import pandas as pd
import imgaug.augmenters as iaa

import sys

ROOT_DIR = os.path.abspath("Resnet/")

sys.path.append(ROOT_DIR)

from resnet.config import Config
from resnet.model import Dataset
from resnet.model import Resnet

"""create Config file with your specific configuration by
 by override Config class"""


class FashionConfig(Config):
    NUMBER_OF_CLASSES = 20

    BATCH_SIZE = 30

    DATA_FRAME_FILE_NAME = 'images.csv'

    LEARNING_RATE = 0.0001


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


def train_phase1(image_dir, base_dir, train_data_filename):
    config = FashionConfig()
    train_data_filename = os.path.join(base_dir, train_data_filename)
    with open(train_data_filename) as f:
        train_data = json.load(f)
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

    argumentation = iaa.SomeOf(1, [
        iaa.AdditiveGaussianNoise(scale=0.15 * 255),
        iaa.Identity(),
        iaa.MotionBlur(k=18),
        iaa.Fliplr(),
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.CropToFixedSize(width=config.IMAGE_INPUT_SHAPE[0], height=config.IMAGE_INPUT_SHAPE[0])
    ], random_order=True)

    result = resnet.train(trainDataset, valDataset, layer='all', augumentation=argumentation)
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
        ckpt_file = os.path.join(base_dir, config.LOG_DIR_NAME, ckpt_file)
    return resnet.evaluate(testDataset, ckpt_file)


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
