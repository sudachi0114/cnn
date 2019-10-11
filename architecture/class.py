
# よりプログラムを綺麗に書くための設計
#   class を使う編

import os, pickle
import numpy as np
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator


class myMobilenetV2():

    def __init__(self):
        self.INPUT_SIZE = 224
        self.TARGET_SIZE = (self.INPUT_SIZE, self.INPUT_SIZE)
        self.CHANNEL = 3
        self.INPUT_SHAPE = (self.INPUT_SIZE, self.INPUT_SIZE, self.CHANNEL)
        self.batch_size = 10
        self.dirs = {}

        current_location = os.path.abspath(__file__)
        cwd, base_name = os.path.split(current_location)
        file_name, _ = os.path.splitext(base_name)
        self.file_name = file_name
        print("current location: ", cwd, ", this file: ", file_name)

        self.dirs['cnn_dir'] = os.path.dirname(cwd)
        self.dirs['data_dir'] = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_mid300")
        self.dirs['train_dir'] = os.path.join(self.dirs['data_dir'], "train")
        self.dirs['validation_dir'] = os.path.join(self.dirs['data_dir'], "validation")

        self.dirs['log_dir'] = os.path.join(cwd, "log")
        os.makedirs(self.dirs['log_dir'], exist_ok=True)
        self.dirs['child_log_dir'] = os.path.join(self.dirs['log_dir'], file_name)
        os.makedirs(self.dirs['child_log_dir'], exist_ok=True)
        print("----- ALL DIRECTORY DEPENDENCY WAS VALIDATED -----")

        self.epochs = 30

    def validation_data_iterator(self):
        validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
        validation_generator = validation_datagen.flow_from_directory(self.dirs['validation_dir'],
                                                                      target_size=self.TARGET_SIZE,
                                                                      batch_size=self.batch_size,
                                                                      class_mode='binary')
        return validation_generator

    def train_data_iterator(self):
        train_datagen = ImageDataGenerator(rescale=1.0/255.0)
        train_generator = train_datagen.flow_from_directory(self.dirs['train_dir'],
                                                            target_size=self.TARGET_SIZE,
                                                            batch_size=self.batch_size,
                                                            class_mode='binary')
        return train_generator

    def train_data_stocker(self):
        x_train, y_train = [], []

        train_generator = self.train_data_iterator()
        
        train_iter_num = train_generator.n//self.batch_size

        for i in range(train_iter_num):
            tmp_x_train, tmp_y_train = next(train_generator)
            if i == 0:
                x_train = tmp_x_train
                y_train = tmp_y_train
            else:
                x_train = np.vstack((x_train, tmp_x_train))
                y_train = np.hstack((y_train, tmp_y_train))

        return x_train, y_train

    
    def create_model(self):
        base_model = MobileNetV2(input_shape=self.INPUT_SHAPE,
                                 weights='imagenet',
                                 include_top=False)
        model = Sequential()
        # 設計は沈さんのモデルを拝借 (activites/20190718)
        model.add(base_model)
        #model.add(GlobalAveragePooling2D())
        # activites/20190712 のものも採用
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # 2値分類の時は sigmoid を選択する

        model.compile(optimizer=Adam(lr=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        return model

    def train_save(self):
        model = self.create_model()

        validation_generator = self.validation_data_iterator()
        train_generator = self.train_data_iterator()

        steps_per_epoch = train_generator.n//self.batch_size
        validation_steps = validation_generator.n//self.batch_size
        print(steps_per_epoch, " [steps / epoch]")
        print(validation_steps, " (validation steps)")

        history = model.fit_generator(train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=self.epochs,
                                      validation_data=validation_generator,
                                      validation_steps=validation_steps,
                                      verbose=1)
        # save model & weights -----
        model_file = os.path.join(self.dirs['child_log_dir'], '{}_model.h5'.format(self.file_name))
        model.save(model_file)

        # save history -----
        history_file = os.path.join(self.dirs['child_log_dir'], '{}_hisoty.pkl'.format(self.file_name))
        with open(history_file, 'wb') as p:
            pickle.dump(history.history, p)

        print("export logs in ... ", self.dirs['child_log_dir'])

        

if __name__ == '__main__':

    mymnv2 = myMobilenetV2()
    mymnv2.train_save()
