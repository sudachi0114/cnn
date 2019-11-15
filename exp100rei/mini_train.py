
import os, sys
sys.path.append(os.pardir)

import argparse

parser = argparse.ArgumentParser(description="ミニマル学習プログラム")

parser.add_argument("--debug", type=int, choices=[1, 2, 3],
                    help="デバッグモードで実行 (Level を 1 ~ 3 の整数値で選択) ")

args = parser.parse_args()

debug_lv = args.debug
if debug_lv == None:
    debug_lv = 0


import numpy as np

import tensorflow as tf
#from keras.backend import tensorflow_backend
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
sess = tf.Session(config=config)
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
K.set_session(sess)

#tensorflow_backend.set_session(session)

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator

#from keras.callbacks import EarlyStopping
#from keras.preprocessing.image import ImageDataGenerator

cwd = os.getcwd()

log_dir = os.path.join(cwd, "mini_log")
#os.makedirs(log_dir, exist_ok=True)

train_dir = os.path.join(cwd, "experiment_0", "train")
validation_dir = os.path.join(cwd, "experiment_0", "validation")
test_dir = os.path.join(cwd, "experiment_0", "test")

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


mh = ModelHandler(224, 3)

model = mh.buildTlearnModel(base='mnv1')

model.summary()


history = model.fit(train_data,
                    train_label,
                    batch_size=10,
                    epochs=30,
                    validation_data=(validation_data, validation_label),
                    verbose=1)

val_accs = history.history['val_accuracy']

print(val_accs[len(val_accs)-1])

print("\npredict sequence...")

pred = model.predict(test_data,
                     #test_label,
                     batch_size=10,
                     verbose=1)

print("result: ", pred)

print("\nevaluate sequence...")

eval_res = model.evaluate(test_data,
                          test_label,
                          batch_size=10,
                          verbose=1)

print("result score: ", eval_res[1])
