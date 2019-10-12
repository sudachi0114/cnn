
# よりプログラムを綺麗に書くための設計
#   import を使ってファイルを分割する編

import os, pickle
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

from model import create_model
from datagen import train_data_iterator, validation_data_iterator

    
def train_save(epochs=30):

    model = create_model()

    validation_generator = validation_data_iterator(validation_dir)
    train_generator = train_data_iterator(train_dir)

    data_checker, label_checker = next(train_generator)
    print("data_checker.shape: ", data_checker.shape)
    print("labal_checker.shape: ", label_checker.shape)

    batch_size = len(data_checker)
    print("batch_size: ", batch_size)

    
    steps_per_epoch = train_generator.n//batch_size
    validation_steps = validation_generator.n//batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  verbose=1)
    # save model & weights -----
    model_file = os.path.join(child_log_dir, '{}_model.h5'.format(file_name))
    model.save(model_file)

    # save history -----
    history_file = os.path.join(child_log_dir, '{}_history.pkl'.format(file_name))
    with open(history_file, 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ... ", child_log_dir)

        

if __name__ == '__main__':

    current_location = os.path.abspath(__file__)
    cwd, base_name = os.path.split(current_location)
    file_name, _ = os.path.splitext(base_name)
    print("current location: ", cwd, ", this file: ", file_name)

    cnn_dir = os.path.dirname(os.path.dirname(cwd))
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_mid300")
    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "validation")

    log_dir = os.path.join(cwd, "log")
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, file_name)
    os.makedirs(child_log_dir, exist_ok=True)
    print("----- ALL DIRECTORY DEPENDENCY WAS VALIDATED -----")

    #epochs = 30

    train_save()
