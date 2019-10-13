
# 2値分類 のプログラムにする。
#   dogs vs cats 12500*2 枚 全てを学習

# ----- import -----
import os

import tensorflow as tf
# GPU を用いるときの tf の session の設定
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from model import build_model
from data_generator import DataGenerator

def main():

    # directory defin -----
    current_location = os.path.abspath(__file__)
    cwd, base_name = os.path.split(current_location)
    file_name, _ = os.path.splitext(base_name)
    print("current location : ", cwd, ", this file : ", file_name)

    cnn_dir = os.path.dirname(cwd)
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_full")
    train_dir = os.path.join(data_dir, "train")
    print("train data is in ... ", train_dir)
    validation_dir = os.path.join(data_dir, "validation")
    print("validation data is in ... ", validation_dir)

    # make log dir
    log_dir = os.path.join(cwd, 'log')
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, "{}_log".format(file_name))
    os.makedirs(child_log_dir, exist_ok=True)


    generator = DataGenerator()
    print("batch size before:", generator.BATCH_SIZE)
    generator.BATCH_SIZE = 100
    print("batch size after:", generator.BATCH_SIZE)
    
    train_generator = generator.Generator(train_dir)
    validation_generator = generator.Generator(validation_dir)
    
    data_checker, label_checker = next(train_generator)
    print("data shape : ", data_checker.shape)
    print("label shape : ", label_checker.shape)

    batch_size = data_checker.shape[0]
    input_size = data_checker.shape[1]
    ch = data_checker.shape[3]
        
    model = build_model(input_size, ch)

    model.summary()

    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=30,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  verbose=1)

    # save model & weights
    model.save(os.path.join(child_log_dir, '{}_model.h5'.format(file_name)))

    # save history
    import pickle
    with open(os.path.join(child_log_dir, '{}_history.pkl'.format(file_name)), 'wb') as p:
        pickle.dump(history.history, p)
                                  
    print("export logs in ", child_log_dir)


if __name__ == '__main__':
    main()
