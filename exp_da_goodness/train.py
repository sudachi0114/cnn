
# Data Argumentation の効果検証用プログラム
#   train 担当

import os, sys, pickle
sys.path.append(os.pardir)
import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from keras.callbacks import EarlyStopping

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator


cwd = os.getcwd()
print("current location : ", cwd)

cnn_dir = os.path.dirname(cwd)
validation_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller", "validation")




def train(aug_no, model_mode='mymodel', set_epochs=10, do_es=False):


    train_dir = os.path.join(cwd, "da_concat_{}".format(aug_no))

    train_data, train_label = inputDataCreator(train_dir,
                                               224,
                                               normalize=True,
                                               one_hot=True)

    validation_data, validation_label = inputDataCreator(validation_dir,
                                                         224,
                                                         normalize=True,
                                                         one_hot=True)

    print("train data shape : ", train_data.shape)
    print("train label shape : ", train_label.shape)


    INPUT_SIZE = train_data.shape[1]
    print("INPUT_SIZE: ", INPUT_SIZE)

    CHANNEL = train_data.shape[3]
    print("set channel : ", CHANNEL)

    batch_size = 10
    print("set batch_size : ", batch_size)


    mh = ModelHandler(INPUT_SIZE, CHANNEL)

    if model_mode == 'mymodel':
        model = mh.buildMyModel()
    elif model_mode == 'tlearn':
        model = mh.buildTlearnModel(base='mnv1')

    model.summary()

    if do_es:
        es = EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           mode='auto')
        es = [es]
    else:
        es = None


    history = model.fit(train_data,
                        train_label,
                        batch_size=batch_size,
                        epochs=set_epochs,
                        validation_data=(validation_data, validation_label),
                        callbacks=es,
                        verbose=1)
    # make log dir -----
    if do_es:
        log_dir = os.path.join(cwd, 'log_with_es')
    else:
        log_dir = os.path.join(cwd, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    
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

    import argparse

    parser = argparse.ArgumentParser("DA の数を増やして DA 自体の良さを検証")

    parser.add_argument("--earlystopping", "-es", action='store_true',
                        help='学習時に EarlyStopping 機能を ON にする。')

    args = parser.parse_args()

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
            train(aug_no=i,
                  model_mode=model_mode,
                  set_epochs=set_epochs,
                  do_es=args.earlystopping)
    print("All task has done !!")
