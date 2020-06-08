
# Data Argumentation の効果検証用プログラム (比較)
#   DA: する/しない * NW: suzuki/Mn(v1)

import os, sys, pickle
sys.path.append(os.pardir)
import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from model_handler import ModelHandler
from da_handler import DaHandler



def train(mode, base, set_epochs=30):

    print("===== mode: {} | base: {} =====".format(mode, base))

    cwd = os.getcwd()
    print("current working dir: ", cwd)

    cnn_dir = os.path.dirname(cwd)

    if mode == 'integrated':
        data_dir = os.path.join(cnn_dir, "dogs_vs_cats_integrated")
    elif mode == 'native':
        data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    train_dir = os.path.join(data_dir, "train")  # global
    validation_dir = os.path.join(data_dir, "validation")  # global
    print("train data is in ... ", train_dir)
    print("validation data is in ...", validation_dir)

    # make log dir -----
    log_dir = os.path.join(cwd, 'comp_log')
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, "{}_{}".format(mode, base))
    os.makedirs(child_log_dir, exist_ok=True)


    dh = DaHandler()
    train_generator = dh.dataGeneratorFromDir(target_dir=train_dir)
    validation_generator = dh.dataGeneratorFromDir(target_dir=validation_dir)

    data_checker, label_checker = next(train_generator)

    print("data_checker shape : ", data_checker.shape)
    print("label_checker shape : ", label_checker.shape)


    INPUT_SIZE = data_checker.shape[1]
    print("INPUT_SIZE: ", INPUT_SIZE)

    CHANNEL = data_checker.shape[3]
    print("set channel : ", CHANNEL)

    batch_size = data_checker.shape[0]
    print("batch_size : ", batch_size)

    mh = ModelHandler(INPUT_SIZE, CHANNEL)
    
    if base == 'mymodel':
        model = mh.buildMyModel()
    elif base == 'mnv1':
        model = mh.buildTlearnModel(base='mnv1')

    model.summary()

    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")

    if mode == 'native':
        set_epochs *= 2

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=set_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  verbose=1)

    # save model & weights
    model_file = os.path.join(child_log_dir, '{}_{}_model.h5'.format(mode, base))
    model.save(model_file)

    # save history
    history_file = os.path.join(child_log_dir, '{}_{}_history.pkl'.format(mode, base))
    with open(history_file, 'wb') as p:
        pickle.dump(history.history, p)

    print("\nexport logs in ", child_log_dir)

    # return 処理 -----
    acc_list = history.history['accuracy']
    last_acc = acc_list[len(acc_list)-1]
    print("\nlast accuracy: ", last_acc)
    val_acc_list = history.history['val_accuracy']
    last_val_acc = val_acc_list[len(val_acc_list)-1]
    print("last validation accuracy: ", last_val_acc)

    return last_acc, last_val_acc

    
if __name__ == '__main__':

    mode_list = ['native', 'integrated']
    base_list = ['mymodel', 'mnv1']
    
    aggrigation = {}

    for mode in mode_list:
        for base in base_list:
            acc, val_acc = train(mode=mode, base=base)
            aggrigation['{}_{}'.format(mode, base)] = [acc, val_acc]
