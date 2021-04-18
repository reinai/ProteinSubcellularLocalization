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
from dataset_util import DataUtil
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class CustomInceptionModel:

    def __init__(self, input_shape, output_size, warmup_epochs, regular_epochs, batch_size, checkpoint_path,
                 path_to_train, image_size):
        self.input_shape = input_shape
        self.output_size = output_size
        self.warmup_epochs = warmup_epochs
        self.regular_epochs = regular_epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.data_util = DataUtil(path_to_train=path_to_train)
        self.image_size = image_size

    def create_inception_model(self):
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
        model = Model(input_tensor, output)
        return model

    def train_inception_model(self):
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

        # create train and valid datagens
        train_generator = self.data_util.create_train(train_indexes, self.batch_size, self.input_shape, augment=True)
        validation_generator = self.data_util.create_train(valid_indexes, 32, self.input_shape, augment=False)

        # warm up model
        model = self.create_inception_model()
        for layer in model.layers:
            layer.trainable = False
        for i in range(1, 7):
            model.layers[-1 * i].trainable = True

        model.compile(loss='binary_crossentropy', optimizer=Adam(1e-03), metrics=['acc'])
        # model.summary()
        model.fit_generator(train_generator, steps_per_epoch=np.ceil(float(len(train_indexes))/float(self.batch_size)),
                            validation_data=validation_generator,
                            validation_steps=np.ceil(float(len(valid_indexes))/float(self.batch_size)),
                            epochs=self.warmup_epochs, verbose=1)

        # train all layers
        for layer in model.layers:
            layer.trainable = True
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
        model.fit_generator(train_generator, steps_per_epoch=np.ceil(float(len(train_indexes))/float(self.batch_size)),
                            validation_data=validation_generator,
                            validation_steps=np.ceil(float(len(valid_indexes)) / float(self.batch_size)),
                            epochs=self.regular_epochs, verbose=1, callbacks=callbacks_list)
