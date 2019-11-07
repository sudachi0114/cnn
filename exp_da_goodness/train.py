
# Data Argumentation の効果検証用プログラム
#   train 担当

import os, sys, pickle
sys.path.append(os.pardir)
import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from utils.model_handler import ModelHandler
from utils.da_handler import DaHandler


cwd = os.getcwd()
print("current location : ", cwd)

cnn_dir = os.path.dirname(cwd)
validation_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller", "validation")

# make log dir -----
log_dir = os.path.join(cwd, 'log')
os.makedirs(log_dir, exist_ok=True)



def train(aug_no, model_mode='mymodel', set_epochs=10):

    dah = DaHandler()

    train_dir = os.path.join(cwd, "da_concat_{}".format(aug_no))
    train_generator = dah.dataGeneratorFromDir(target_dir=train_dir)

    validation_generator = dah.dataGeneratorFromDir(target_dir=validation_dir)

    data_checker, label_checker = next(train_generator)

    print("data_checker shape : ", data_checker.shape)
    print("label_checker shape : ", label_checker.shape)


    INPUT_SIZE = data_checker.shape[1]
    print("INPUT_SIZE: ", INPUT_SIZE)

    CHANNEL = data_checker.shape[3]
    print("set channel : ", CHANNEL)

    batch_size = data_checker.shape[0]
    print("batch_size : ", batch_size)


    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")

    mh = ModelHandler(INPUT_SIZE, CHANNEL)

    if model_mode == 'mymodel':
        model = mh.buildMyModel()
    elif model_mode == 'tlearn':
        model = mh.buildTlearnModel(base='mnv1')

    model.summary()


    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=set_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  verbose=1)


    child_log_dir = os.path.join(log_dir, '{}_{}'.format(aug_no, model_mode))
    os.makedirs(child_log_dir, exist_ok=True)

    # save model & weights
    model_file = os.path.join(child_log_dir, '{}_{}_model.h5'.format(aug_no, model_mode))
    model.save(model_file)

    # save history
    history_file = os.path.join(child_log_dir, '{}_{}_history.pkl'.format(aug_no, model_mode))
    with open(history_file, 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)


if __name__ == '__main__':

    picked_aug_list = ["rotation", "hflip", "gnoise", "invert", "native"]

    model_mode_list = ['mymodel', 'tlearn']

    otameshi_train = [1, 31]
    #for i in range(1, 2**len(picked_aug_list)):
    for i in otameshi_train:
        if i == 1:
            set_epochs = 100
        elif i == 31:
            set_epochs = 20
        for model_mode in model_mode_list:
            print("========== auged No: {} | model: {} ==========".format(i, model_mode))
            train(aug_no=i, model_mode=model_mode, set_epochs=set_epochs)
    print("All task has done !!")
