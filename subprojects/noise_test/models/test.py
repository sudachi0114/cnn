
# transfer learning with finetuning

import os, sys
sys.path.append(os.pardir)

import time
import numpy as np
np.random.seed(seed=114)
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import backend as K
config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.5
sess = tf.Session(config=config)
K.set_session(sess)

print("TensorFlow version is ", tf.__version__)
print("Keras version is ", keras.__version__)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix

from utils.model_handler import ModelHandler


# define -----
batch_size = 50
input_size = 224
channel = 3
target_size = (input_size, input_size)
input_shpe = (input_size, input_size, channel)
set_epochs = 40



def main():

    cwd = os.getcwd()
    sub_prj = os.path.dirname(cwd)
    """
    sub_prj_root = os.path.dirname(sub_prj)
    prj_root = os.path.dirname(sub_prj_root)

    """
    data_dir = os.path.join(sub_prj, "datasets")

    # data_src = os.path.join(data_dir, "medium_721")
    data_src = os.path.join(data_dir, "small_721")
    print("\ndata source: ", data_src)

    use_da_data = False
    if use_da_data:
        test_dir = os.path.join(data_src, "test_with_aug")
    else:
        test_dir = os.path.join(data_src, "test")

    print("test_dir: ", test_dir)


    # data load ----------
    data_gen = ImageDataGenerator(rescale=1./255)

    test_generator = data_gen.flow_from_directory(test_dir,
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  class_mode='categorical')

    # load model ----------
    models_dir = os.path.join(sub_prj, "outputs", "models")
    model_file = "finetune_model.h5"  # "mymodel_auged.h5"
    model_location = os.path.join(models_dir, model_file)
    print("\nmodel location: ", model_location)
    model = load_model(model_location, compile=True)

    model.summary()

    print("\ntest sequence...")
    test_steps = test_generator.n // batch_size
    eval_res = model.evaluate_generator(test_generator,
                                        steps=test_steps,
                                        verbose=1)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])

    # confusion matrix -----
    print("\nconfusion matrix")
    pred = model.predict_generator(test_generator,
                                   steps=test_steps,
                                   verbose=3)

    test_label = []
    for i in range(test_steps):
        tmp_data, tmp_tl = next(test_generator)
        if i == 0:
            test_label = tmp_tl
        else:
            test_label = np.vstack((test_label, tmp_tl))

        if False:  # display
            plt.figure(figsize=(12, 6))
            for j in range(batch_size):
                plt.subplot(5, 10, j+1)
                plt.imshow(tmp_data[j])
                plt.title("{}".format(tmp_tl[j]))
                plt.axis(False)

            plt.show()


    idx_label = np.argmax(test_label, axis=-1)  # one_hot => normal
    idx_pred = np.argmax(pred, axis=-1)  # 各 class の確率 => 最も高い値を持つ class

    cm = confusion_matrix(idx_label, idx_pred)

    # Calculate Precision and Recall
    tn, fp, fn, tp = cm.ravel()


    print("  | T  | F ")
    print("--+----+---")
    print("N | {} | {}".format(tn, fn))
    print("--+----+---")
    print("P | {} | {}".format(tp, fp))



if __name__ == '__main__':
    main()
