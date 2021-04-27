import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from inceptionnet.dataset_util import DataUtil
import numpy as np
from inceptionnet.macro_f1 import macro_f1_score
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class CustomInceptionModel:
    """
    Main class for the InceptionV3 model, having all function that is needed for the model to be trained and tested
    """

    def __init__(self, input_shape, output_size, warmup_epochs, regular_epochs, batch_size, checkpoint_path,
                 path_to_train, path_to_test, image_size):
        """
        Constructor for the custom InceptionV3 model

        :param input_shape: input shape for the model
        :param output_size: output size of the model, number of output classes
        :param warmup_epochs: number of training epochs used to train only additional layers added
        :param regular_epochs: number of training epochs used to train whole model
        :param batch_size: size of a batch for training a model to implement mini batch gradient
        :param checkpoint_path: path where the checkpoint with the model weights is saved
        :param path_to_train: path for the train set
        :param path_to_test: path for the test set
        :param image_size: size of an image (it is a square image so it mean length/width of an image)
        """

        self.input_shape = input_shape
        self.output_size = output_size
        self.warmup_epochs = warmup_epochs
        self.regular_epochs = regular_epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.data_util = DataUtil(path_to_train=path_to_train, path_to_test=path_to_test)
        self.image_size = image_size
        self.model = None

    def create_inception_model(self):
        """
        Function to create custom InceptionV3 model with additional layers on the top instead of the default ones
        Image of the models architecture can be found in ./model_image/custom_inceptionV3_model.png
        """

        input_tensor = Input(shape=self.input_shape)
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=self.input_shape)
        bn = BatchNormalization()(input_tensor)
        x = base_model(bn)
        x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.output_size, activation='sigmoid')(x)
        self.model = Model(input_tensor, output)

    def train_inception_model(self):
        """
        Main function to train a custom InceptionV3 model
        """

        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min', save_weights_only=True)
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto',
                                                 epsilon=0.0001)
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=6)
        callbacks_list = [checkpoint, early_stopping, reduce_lr_on_plateau]

        # split data into train, valid
        self.data_util.preprocess_train_data()
        indexes = np.arange(self.data_util.train_dataset.shape[0])
        np.random.shuffle(indexes)
        train_indexes, valid_indexes = train_test_split(indexes, test_size=0.05, random_state=8)

        # create train and valid data generators
        train_generator = self.data_util.create_train(train_indexes, self.batch_size, self.input_shape, augment=True)
        validation_generator = self.data_util.create_train(valid_indexes, 32, self.input_shape, augment=False)

        # instantiate a custom inception model
        self.create_inception_model()

        # warm up model by training only newly added layers
        for layer in self.model.layers:
            layer.trainable = False
        for i in range(1, 7):
            self.model.layers[-1 * i].trainable = True
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(1e-03), metrics=[macro_f1_score])
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=np.ceil(float(len(train_indexes))/float(self.batch_size)),
                                 validation_data=validation_generator,
                                 validation_steps=np.ceil(float(len(valid_indexes))/float(self.batch_size)),
                                 epochs=self.warmup_epochs, verbose=1)

        # train all layers of the model
        for layer in self.model.layers:
            layer.trainable = True
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=[macro_f1_score])
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=np.ceil(float(len(train_indexes))/float(self.batch_size)),
                                 validation_data=validation_generator,
                                 validation_steps=np.ceil(float(len(valid_indexes)) / float(self.batch_size)),
                                 epochs=self.regular_epochs, verbose=1, callbacks=callbacks_list)

    def test_inception_model(self):
        """
        Function to test the performance of our custom inception model on test set and saving the results in the csv
        file
        """

        # importing an empty csv file that will be filled with the predicted results
        predicted_csv = pd.read_csv('../dataset/empty_test.csv')

        # instantiate a custom inception model
        self.create_inception_model()
        self.model.load_weights(self.checkpoint_path)

        self.data_util.preprocess_test_data()
        test_images, test_labels = self.data_util.create_test(self.input_shape)
        predicted = []
        for iter in range(len(test_images)):
            score_predict = self.model.predict(test_images[iter][np.newaxis])[0]
            label_predict = np.arange(28)[score_predict >= 0.2]
            str_predict_label = ' '.join(str(label) for label in label_predict)
            predicted.append(str_predict_label)

        # saving the predicted results in the csv file
        predicted_csv['Predicted'] = predicted
        predicted_csv.to_csv('../dataset/predicted_InceptionV3.csv', index=False)

    def calculate_predicted_macro_f1_score(self, predicted_csv_path):
        """
        Calculating the Macro F1 score based on the predicted results from the csv file and the true ground labels of
        the test set

        :param predicted_csv_path: path for the predicted results saved in the csv file
        :return: macro f1 score between predicted and true labels
        """

        self.data_util.preprocess_test_data()
        test_labels = self.data_util.get_test_labels()
        predicted_labels = np.zeros((len(test_labels), 28))
        predicted_csv = pd.read_csv(predicted_csv_path)
        for index, labels in enumerate(predicted_csv['Predicted'].str.split(' ')):
            try:
                labels_new = np.array([int(label) for label in labels])
            except:
                labels_new = []
            predicted_labels[index][labels_new] = 1
        return macro_f1_score(y_true=test_labels, y_predicted=predicted_labels)
