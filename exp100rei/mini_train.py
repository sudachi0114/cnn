
import os, sys
sys.path.append(os.pardir)

"""
import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)
"""

import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator

from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

cwd = os.getcwd()

log_dir = os.path.join(cwd, "mini_log")
#os.makedirs(log_dir, exist_ok=True)

train_dir = os.path.join(cwd, "experiment_0", "train")
validation_dir = os.path.join(cwd, "experiment_0", "validation")
test_dir = os.path.join(cwd, "experiment_0", "test")

print("\ncreate train data")
train_data, train_label = inputDataCreator(train_dir,
                                           224,
                                           normalize=True)


print("train_data: ", train_data.shape)
print("train_label: ", train_label.shape)

print(train_data[0])
print(train_label)

print("\ncreate validation data")
validation_data, validation_label = inputDataCreator(validation_dir,
                                                     224,
                                                     normalize=True)

print("validation_data: ", validation_data.shape)
print("validation_label: ", validation_label.shape)

print(validation_data[0])
print(validation_label)


print("\ncreate test data")
test_data, test_label = inputDataCreator(test_dir,
                                         224,
                                         normalize=True)

print("test_data: ", test_data.shape)
print("test_label: ", test_label.shape)

print(test_data[0])
print(test_label)



mh = ModelHandler(224, 3)

model = mh.buildTlearnModel(base='mnv1')

model.summary()


history = model.fit(train_data,
                    train_label,
                    batch_size=10,
                    epochs=40,
                    validation_data=(validation_data, validation_label),
                    verbose=1)

val_accs = history.history['val_accuracy']

print(val_accs[len(val_accs)-1])
