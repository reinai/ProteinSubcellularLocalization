import cv2
import PIL
import os
import sys
import skimage.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from imgaug import augmenters as iaa
from sklearn.utils import class_weight, shuffle
import warnings
warnings.filterwarnings("ignore")


class DataUtil:
    """
    Class that is used for data manipulation, concretely to preprocess train and test data
    """

    def __init__(self, path_to_train, path_to_test):
        """
        Constructor for the DataUtil class

        :param path_to_train: path for the train set
        :param path_to_test: path for the test set
        """

        self.path_to_train = path_to_train
        self.path_to_test = path_to_test
        self.train_dataset = []
        self.test_dataset = []

    def preprocess_train_data(self):
        """
        Function to read train csv file and put that data into more meaningful data structure
        """

        train_data = pd.read_csv(self.path_to_train + '/train.csv')
        for name, labels in zip(train_data['Id'], train_data['Target'].str.split(' ')):
            self.train_dataset.append({
                'path': os.path.join(self.path_to_train + '/train/', name),
                'labels': np.array([int(label) for label in labels])})
        self.train_dataset = np.array(self.train_dataset)

    def preprocess_test_data(self):
        """
        Function to read test csv file and put that data into more meaningful data structure
        """

        test_data = pd.read_csv(self.path_to_test + '/test.csv')
        for name, labels in zip(test_data['Id'], test_data['Predicted'].str.split(' ')):
            # because there are rows that have predicted 0 labels, it is needed to do this part like this
            try:
                labels_new = np.array([int(label) for label in labels])
            except:
                # there is no predicted label
                labels_new = []
            self.test_dataset.append({
                'path': os.path.join(self.path_to_test + '/test/', name),
                'labels': labels_new })
        self.test_dataset = np.array(self.test_dataset)

    def get_test_labels(self):
        """
        Function to return true ground labels for the test set

        :return: true ground labels from test set
        """

        test_labels = np.zeros((len(self.test_dataset), 28))
        for iter in range(len(self.test_dataset)):
            test_labels[iter][self.test_dataset[iter]['labels']] = 1
        return test_labels

    def create_train(self, dataset_index, batch_size, shape, augment=True):
        """
        Function that is generator to create batch for the training purpose, with an option to augment the data

        :param dataset_index: starting index that represents from which index of training dataset the batch starts
        :param batch_size: size of one training batch
        :param shape: shape of an image needed for the model
        :param augment: boolean value that is used to augment the data or not
        :return: yielding (because it is a generator) batch images and batch labels
        """

        while True:
            part_of_dataset = self.train_dataset[dataset_index]
            part_of_dataset = shuffle(part_of_dataset, random_state=77)
            for start_of_batch in range(0, len(part_of_dataset), batch_size):
                end_of_batch = min(start_of_batch + batch_size, len(part_of_dataset))
                batch_images = []
                train_batch = part_of_dataset[start_of_batch:end_of_batch]
                batch_labels = np.zeros((len(train_batch), 28))
                for i in range(len(train_batch)):
                    image = self.load_rgb_image(train_batch[i]['path'], shape)
                    if augment:
                        # augment an image
                        image = self.augment_image(image)
                    # normalizing an image
                    batch_images.append(image / 255.)
                    # setting labels to 1 and non-labels to 0
                    batch_labels[i][train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def create_test(self, shape):
        """
        Function to create a full test set that will go through the same preprocessing as train set

        :param shape: shape of an image needed for the model
        :return: test set images and labels
        """

        test_labels = np.zeros((len(self.test_dataset), 28))
        test_images = []
        for iter in range(len(self.test_dataset)):
            image = self.load_rgb_image(self.test_dataset[iter]['path'], shape)
            # normalizing an image
            test_images.append(image / 255.)
            # setting labels to 1 and non-labels to 0
            test_labels[iter][self.test_dataset[iter]['labels']] = 1
        return np.array(test_images, np.float32), test_labels

    @staticmethod
    def load_rgb_image(path, shape):
        """
        Function to load a RGB channeled image in a way to combine these three channel for the concrete id of the image

        :param path: path for the image with an id of the image
        :param shape: shape of the image that we want to be an output
        :return: RGB image
        """

        image_red_channel = Image.open(path + '_red.png')
        image_green_channel = Image.open(path + '_green.png')
        image_blue_channel = Image.open(path + '_blue.png')
        # there is no yellow channel because its information is encapsulated through green and red channel
        image = np.stack((np.array(image_red_channel), np.array(image_green_channel), np.array(image_blue_channel)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    @staticmethod
    def augment_image(image):
        """
        Function to make a different version of an image through augment techniques, like rotating and flipping

        :param image: input image
        :return: augmented image
        """

        augmented_image = iaa.Sequential([iaa.OneOf([
                iaa.Affine(rotate=0), iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270),
                iaa.Fliplr(0.5), iaa.Flipud(0.5)])], random_order=True, random_state=10)
        augmented_image = augmented_image.augment_image(image)
        return augmented_image
