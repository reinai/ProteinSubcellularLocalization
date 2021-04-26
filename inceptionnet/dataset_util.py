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

    def __init__(self, path_to_train, path_to_test):
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test
        self.train_dataset = []
        self.test_dataset = []

    def preprocess_train_data(self):
        train_data = pd.read_csv(self.path_to_train + '/train.csv')
        for name, labels in zip(train_data['Id'], train_data['Target'].str.split(' ')):
            self.train_dataset.append({
                'path': os.path.join(self.path_to_train + '/train/', name),
                'labels': np.array([int(label) for label in labels])})
        self.train_dataset = np.array(self.train_dataset)

    def preprocess_test_data(self):
        test_data = pd.read_csv(self.path_to_test + '/test.csv')
        for name, labels in zip(test_data['Id'], test_data['Predicted'].str.split(' ')):
            try:
                labels_new = np.array([int(label) for label in labels])
            except:
                labels_new = []
            self.test_dataset.append({
                'path': os.path.join(self.path_to_test + '/test/', name),
                'labels': labels_new })
        self.test_dataset = np.array(self.test_dataset)

    def get_test_labels(self):
        test_labels = np.zeros((len(self.test_dataset), 28))
        for iter in range(len(self.test_dataset)):
            test_labels[iter][self.test_dataset[iter]['labels']] = 1
        return test_labels

    def create_train(self, dataset_index, batch_size, shape, augment=True):
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
                        image = self.augment_image(image)
                    batch_images.append(image / 255.)
                    batch_labels[i][train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def create_test(self, shape):
        test_labels = np.zeros((len(self.test_dataset), 28))
        test_images = []
        for iter in range(len(self.test_dataset)):
            image = self.load_rgb_image(self.test_dataset[iter]['path'], shape)
            test_images.append(image / 255.)
            test_labels[iter][self.test_dataset[iter]['labels']] = 1
        return np.array(test_images, np.float32), test_labels

    @staticmethod
    def load_rgb_image(path, shape):
        image_red_channel = Image.open(path + '_red.png')
        image_green_channel = Image.open(path + '_green.png')
        image_blue_channel = Image.open(path + '_blue.png')
        image = np.stack((np.array(image_red_channel), np.array(image_green_channel), np.array(image_blue_channel)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    @staticmethod
    def augment_image(image):
        augmented_image = iaa.Sequential([iaa.OneOf([
                iaa.Affine(rotate=0), iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270),
                iaa.Fliplr(0.5), iaa.Flipud(0.5)])], random_order=True, random_state=10)
        augmented_image = augmented_image.augment_image(image)
        return augmented_image
