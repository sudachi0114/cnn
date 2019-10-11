
# よりプログラムを綺麗に書くための設計
#   import を使ってファイルを分割する編

import os
import numpy as np
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator


def validation_data_iterator(validation_dir, INPUT_SIZE=224, batch_size=10):

    TARGET_SIZE = (INPUT_SIZE, INPUT_SIZE)
    
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=TARGET_SIZE,
                                                                  batch_size=batch_size,
                                                                  class_mode='binary')
    return validation_generator

def train_data_iterator(train_dir, INPUT_SIZE=224, batch_size=10):

    TARGET_SIZE = (INPUT_SIZE, INPUT_SIZE)
    
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=TARGET_SIZE,
                                                        batch_size=batch_size,
                                                        class_mode='binary')
    return train_generator

def train_data_stocker(batch_size=10):
    x_train, y_train = [], []

    train_generator = train_data_iterator()

    train_iter_num = train_generator.n//batch_size

    for i in range(train_iter_num):
        tmp_x_train, tmp_y_train = next(train_generator)
        if i == 0:
            x_train = tmp_x_train
            y_train = tmp_y_train
        else:
            x_train = np.vstack((x_train, tmp_x_train))
            y_train = np.hstack((y_train, tmp_y_train))

    return x_train, y_train


if __name__ == '__main__':

    current_location = os.path.abspath(__file__)
    cwd, base_name = os.path.split(current_location)
    file_name, _ = os.path.splitext(base_name)
    print("current location: ", cwd, ", this file: ", file_name)

    cnn_dir = os.path.dirname(os.path.dirname(cwd))
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_mid300")
    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "validation")

    print("----- ALL DIRECTORY DEPENDENCY WAS VALIDATED -----")

    #batch_size = 10

    train_generator = train_data_iterator(train_dir)
    validation_gatagen = validation_data_iterator(validation_dir)

    data_checker, label_checker = next(train_generator)
    print("data_checker.shape: ", data_checker.shape)
    print("label_checker.shape: ", label_checker.shape)
