
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
validation_data, validation_label = inputDataCreator(validation_dir,
                                                     224,
                                                     normalize=True,
                                                     one_hot=True)

print("validation_data shape: ", validation_data.shape)
print("validation_label shape: ", validation_label.shape)



def train(is_aug, model_mode, set_epochs=100, do_es=False):


    if is_aug:
        train_dir = os.path.join(cwd, "da_concat")
        if do_es == False:
            set_epochs = int(set_epochs/4)        
    else:
        train_dir = os.path.join(cwd, "dogs_vs_cats_auged_native")

    train_data, train_label = inputDataCreator(train_dir,
                                               224,
                                               normalize=True,
                                               one_hot=True)

    print("train_data shape : ", train_data.shape)
    print("train_label shape : ", train_label.shape)


    INPUT_SIZE = train_data.shape[1]
    print("INPUT_SIZE: ", INPUT_SIZE)

    CHANNEL = train_data.shape[3]
    print("set channel : ", CHANNEL)

    batch_size = 10
    print("batch_size : ", batch_size)


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


    child_log_dir = os.path.join(log_dir, '{}_{}'.format(is_aug, model_mode))
    os.makedirs(child_log_dir, exist_ok=True)

    # save model & weights
    model_file = os.path.join(child_log_dir, '{}_{}_model.h5'.format(is_aug, model_mode))
    model.save(model_file)

    # save history
    history_file = os.path.join(child_log_dir, '{}_{}_history.pkl'.format(is_aug, model_mode))
    with open(history_file, 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="いくつかの DA を施す方法の検証")

    parser.add_argument("--earlystopping", "-es", action="store_true",
                        help="学習時に EarlyStopping を ON にする.")

    args = parser.parse_args()

    model_mode_list = ['mymodel', 'tlearn']

    for i in range(2):
        for model_mode in model_mode_list:
            print("========== is_auged : {} | model: {} ==========".format(bool(i), model_mode))
            train(is_aug=i,
                  model_mode=model_mode,
                  do_es=args.earlystopping)
    print("All task has done !!")
