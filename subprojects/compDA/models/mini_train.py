
import os, sys
sys.path.append(os.pardir)

import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator

#from keras.callbacks import EarlyStopping
#from keras.preprocessing.image import ImageDataGenerator


def train(LEARN_PATH, INPUT_SIZE, CHANNEL, BATCH_SIZE, EPOCHS):

    # log_dir = os.path.join(cwd, "mini_log")
    #os.makedirs(log_dir, exist_ok=True)

    # train_dir = os.path.join(LEARN_PATH, "train")
    train_dir = os.path.join(LEARN_PATH, "train_with_aug")
    validation_dir = os.path.join(LEARN_PATH, "validation")
    test_dir = os.path.join(LEARN_PATH, "test")

    print("\ncreate train data")
    train_data, train_label = inputDataCreator(train_dir,
                                               224,
                                               normalize=True,
                                               one_hot=True)


    if debug_lv > 0:
        print("train_data: ", train_data.shape)
        print("train_label: ", train_label.shape)

        if debug_lv > 1:
            print(train_data[0])
            print(train_label)

    print("\ncreate validation data")
    validation_data, validation_label = inputDataCreator(validation_dir,
                                                         224,
                                                         normalize=True,
                                                         one_hot=True)


    if debug_lv > 0:
        print("validation_data: ", validation_data.shape)
        print("validation_label: ", validation_label.shape)

        if debug_lv > 1:
            print(validation_data[0])
            print(validation_label)


    print("\ncreate test data")
    test_data, test_label = inputDataCreator(test_dir,
                                             224,
                                             normalize=True,
                                             one_hot=True)


    if debug_lv > 0:
        print("test_data: ", test_data.shape)
        print("test_label: ", test_label.shape)

        if debug_lv > 1:
            print(test_data[0])
            print(test_label)


    mh = ModelHandler(INPUT_SIZE, CHANNEL)    
    model = mh.buildTlearnModel(base='mnv1')

    model.summary()


    history = model.fit(train_data,
                        train_label,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(validation_data, validation_label),
                        verbose=2)

    accs = history.history['accuracy']
    losses = history.history['loss']
    val_accs = history.history['val_accuracy']
    val_losses = history.history['val_loss']
    print("last val_acc: ", val_accs[len(val_accs)-1])


    print("\npredict sequence...")

    pred = model.predict(test_data,
                         #test_label,
                         batch_size=BATCH_SIZE,
                         verbose=2)

    label_name_list = []
    for i in range(len(test_label)):
        if test_label[i][0] == 1:
            label_name_list.append('cat')
        elif test_label[i][1] == 1:
            label_name_list.append('dog')
        

    #print("result: ", pred)
    df_pred = pd.DataFrame(pred, columns=['cat', 'dog'])
    df_pred['class'] = df_pred.idxmax(axis=1)
    df_pred['label'] = pd.DataFrame(label_name_list, columns=['label'])
    df_pred['collect'] = (df_pred['class'] == df_pred['label'])

    confuse = df_pred[df_pred['collect'] == False].index.tolist()
    collect = df_pred[df_pred['collect'] == True].index.tolist()

    print(df_pred)
    print("\nwrong recognized indeices are ", confuse)
    print("  wrong recognized amount is ", len(confuse))
    print("\ncollect recognized indeices are ", collect)
    print("  collect recognized amount is ", len(collect))
    print("\nwrong rate: ", 100*len(confuse)/len(test_label), " %")



    print("\nevaluate sequence...")
    eval_res = model.evaluate(test_data,
                              test_label,
                              batch_size=BATCH_SIZE,
                              verbose=2)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])

    # ----------
    save_dict = {}
    save_dict['last_loss'] = losses[len(losses)-1]
    save_dict['last_acc'] = accs[len(accs)-1]
    save_dict['last_val_loss'] = val_losses[len(val_losses)-1]
    save_dict['last_val_acc'] = val_accs[len(val_accs)-1]
    save_dict['n_confuse'] = len(confuse)
    save_dict['eval_loss'] = eval_res[0]
    save_dict['eval_acc'] = eval_res[1]

    print(save_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ミニマル学習プログラム")

    parser.add_argument("--debug", type=int, choices=[1, 2, 3],
                        help="デバッグモードで実行 (Level を 1 ~ 3 の整数値で選択) ")

    args = parser.parse_args()

    debug_lv = args.debug
    if debug_lv == None:
        debug_lv = 0

    learn_path = "/home/sudachi/cnn/datasets/1000_721"
    train(LEARN_PATH=learn_path,
          INPUT_SIZE=224,
          CHANNEL=3,
          BATCH_SIZE=10,
          EPOCHS=30)
