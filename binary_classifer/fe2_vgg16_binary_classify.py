
# pre-trained model : VGG16 を用いて画像認識 (特徴量抽出 feature extraction)
#   conv_base の上に分類器を載せてそのままぶん回す方式

import os
import numpy as np

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam


def main(input_size=150, batch_size=10, epochs=30):

    # directory -----
    current_location = os.path.abspath(__file__)
    cwd, base_name = os.path.split(current_location)
    file_name, _ = os.path.splitext(base_name)
    print("current location : ", cwd, ", this file : ", file_name)
    
    cnn_dir = os.path.dirname(cwd)
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    train_dir = os.path.join(data_dir, "train")
    print("train data is in ... ", train_dir)
    validation_dir = os.path.join(data_dir, "validation")
    print("validation data is in ... ", validation_dir)

    log_dir = os.path.join(cwd, "log")
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, "{}_log".format(file_name))
    os.makedirs(child_log_dir, exist_ok=True)
    

    datagen = ImageDataGenerator(rescale=1/255.)

    train_generator = datagen.flow_from_directory(train_dir,
                                                  target_size=(input_size, input_size),
                                                  batch_size=batch_size,
                                                  class_mode='binary')

    validation_generator = datagen.flow_from_directory(validation_dir,
                                                       target_size=(input_size, input_size),
                                                       batch_size=batch_size,
                                                       class_mode='binary')

    data_checker, label_checker = next(train_generator)
    data_shape = data_checker.shape  # (batch_size, width, height, ch)


    # create conv_base -----
    #   using Conv base@VGG16
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(data_shape[1], data_shape[2], data_shape[3]))

    # conv_base のパラメータを凍結
    conv_base.trainable = False

    # model -----
    model = Sequential()

    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])

    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")
    
    
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps)

    # save model in hdf5 file -----
    model.save(os.path.join(child_log_dir, "{}_model.h5".format(file_name)))

    # save history -----
    import pickle
    with open(os.path.join(child_log_dir, "{}_history.pkl".format(file_name)), 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)


if __name__ == '__main__':
    main()
