import sys
import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from PIL import Image
from tqdm import tqdm
import imgaug as ia  # image augmentation
from imgaug import augmenters as iaa
import cv2

import tensorflow as tf

import keras
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, \
    Concatenate, ReLU, LeakyReLU, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import math


BS = 32  # batch size
INITIAL_SEED = 777
IMAGE_SHAPE = (512, 512, 4)
DATA_DIRECTORY = "../dataset"

"""
due to different cost of True Positive vs False Positive, this is the probability threshold to 
predict the class as 'yes'
"""
THRESHOLD = 0.5

VALIDATION_RATIO = 0.1  # 10 % of train dataset as validation set

ia.seed(INITIAL_SEED)


def getDataset(image_folder_path, predictions_csv_path):
    data_information = pd.read_csv(predictions_csv_path)

    paths = []
    labels = []

    for name, lbl in zip(data_information['Id'], data_information['Predicted'].str.split(' ')):
        y = np.zeros(28)

        if type(lbl) is not list:
            if math.isnan(lbl):
                paths.append(os.path.join(image_folder_path, name))
                labels.append(y)
                continue

        for key in lbl:
            y[int(key)] = 1

        paths.append(os.path.join(image_folder_path, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


class ProteinDataGenerator(keras.utils.Sequence):

    def __init__(self, paths, labels, batch_size, shape, shuffle=False, use_cache=False, augment=False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment

        if use_cache:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))

        self.on_epoch_end()

    def __len__(self):
        # number of batches
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))

        # Generate data
        if self.use_cache:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]

        if self.augment:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),  # horizontal flips
                    iaa.Crop(percent=(0, 0.1)),  # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                                  iaa.GaussianBlur(sigma=(0, 0.5))
                                  ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)

        return X, y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R),
            np.array(G),
            np.array(B),
            np.array(Y)), -1)

        im = cv2.resize(im, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        im = np.divide(im, 255)  # normalization
        return im


def macro_F1_score(y_true, y_predicted):
    true_positive = K.sum(K.cast(y_true * y_predicted, 'float'), axis=0)
    false_positive = K.sum(K.cast((1 - y_true) * y_predicted, 'float'), axis=0)
    false_negative = K.sum(K.cast(y_true * (1 - y_predicted), 'float'), axis=0)

    precision = true_positive / (true_positive + false_positive + K.epsilon())
    recall = true_positive / (true_positive + false_negative + K.epsilon())

    macro_f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    macro_f1 = tf.where(tf.math.is_nan(macro_f1), tf.zeros_like(macro_f1), macro_f1)
    return float(K.mean(macro_f1))  # returns tf.tensor representation, thus we need float


def macro_F1_loss(y_true, y_predicted):
    return 1 - macro_F1_score(y_true, y_predicted)


model = load_model('gapnet-pl/models/gapnet.model', custom_objects={'f1': macro_F1_score})
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-03),
    metrics=['acc', macro_F1_score])

print("Model summary:\n")
model.summary()

def test_gapnet_pl():
    image_folder_path = input("\nEnter test image folder path: ")
    our_test_predictions_directory = input("\nEnter file path to our predictions: ")
    test_data_directory = input("\nEnter file path to the test data directory: ")

    # getDataset(image_folder_path, predictions_csv_path)

    paths_test, y_pred = getDataset(image_folder_path, predictions_csv_path=our_test_predictions_directory)
    _, y = getDataset(image_folder_path, predictions_csv_path=test_data_directory)

    testg = ProteinDataGenerator(paths_test, y, BS, IMAGE_SHAPE)

    P = np.zeros((paths_test.shape[0], 28))
    """
    for i in tqdm(range(len(testg))):
        images, labels = testg[i]
        score = model.predict(images)
        P[i * BS:i * BS + score.shape[0]] = score
    """

    print("Macro F1 score: " + str(macro_F1_score(y_true=y, y_predicted=y_pred)))


if __name__ == "__main__":
    test_gapnet_pl()

"""
/home/nikola/Documents/TEST_PROTEIN/test
/home/nikola/Desktop/ds_projekat/ProteinSubcellularLocalization/gapnet-pl/models/predictions_gapnet_pl.csv
/home/nikola/Documents/TEST_PROTEIN/test.csv
"""
