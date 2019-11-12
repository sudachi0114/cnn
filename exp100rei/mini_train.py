
import os, sys
sys.path.append(os.pardir)

import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator

from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

cwd = os.getcwd()

log_dir = os.path.join(cwd, "mini_log")
#os.makedirs(log_dir, exist_ok=True)

mh = ModelHandler(224, 3)

model = mh.buildTlearnModel(base='mnv1')

model.summary()
