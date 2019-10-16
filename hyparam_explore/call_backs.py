
import os, sys
sys.path.append(os.pardir)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from keras.callbacks import EarlyStopping

from model_handler import ModelHandler
from data_handler import DataHandler

def main():

    # data gen -----
    data_handler = DataHandler()
    data_handler.data_purpose = 'train'
    print("data_purpose: ", data_handler.data_purpose)
    train_generator = data_handler.dataGenerator()

    data_checker, _ = next(train_generator)

    batch_size = data_checker.shape[0]
    input_size = data_checker.shape[1]
    ch = data_checker.shape[3]

    data_handler.data_purpose = 'validation'
    print("data_purpose: ", data_handler.data_purpose)
    validation_generator = data_handler.dataGenerator()

    # model -----
    model_handler = ModelHandler(input_size, ch)

    model = model_handler.buildMyModel()

    model.summary()

    current_location = os.path.abspath(__file__)
    cwd, file_ext = os.path.split(current_location)
    file_name, _ = os.path.splitext(file_ext)

    log_dir = os.path.join(cwd, "log")
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, file_name)
    os.makedirs(child_log_dir, exist_ok=True)

    es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    # <keras.callbacks.callbacks.EarlyStopping object at 0x7f98c5a3eef0>

    steps_per_epoch = train_generator.n//batch_size
    print(steps_per_epoch, " [steps / epoch]")
    validation_steps = validation_generator.n//batch_size
    print(validation_steps, " (validation steps)")

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=30,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  callbacks=[es])
    # save model & weights
    model.save(os.path.join(child_log_dir, '{}_model.h5'.format(file_name)))

    # save history
    import pickle
    with open(os.path.join(child_log_dir, '{}_history.pkl'.format(file_name)), 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)    

    
if __name__ == '__main__':

    main()
