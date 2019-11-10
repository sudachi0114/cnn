
import os, sys, argparse, pickle
sys.path.append(os.pardir)

import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)
from keras.callbacks import EarlyStopping

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator



cwd = os.getcwd()

log_dir = os.path.join(cwd, "log")
os.makedirs(log_dir, exist_ok=True)



def main(data_mode, model_mode, no, set_epochs=60, do_es=False):

    base_dir = os.path.join(cwd, "experiment_{}".format(no))

    if data_mode == 'native':
        train_dir = os.path.join(cwd, "experiment_{}".format(no), "train")
    elif data_mode == 'auged':
        auged_train_dir = os.path.join(cwd, "concat_experiment_{}".format(no))

    validation_dir = os.path.join(base_dir, "validation")
    #test_dir = os.path.join(base_dir, "test")


    train_data, train_label = inputDataCreator(train_dir,
                                               224,
                                               normalize=True)

    validation_data, validation_label = inputDataCreator(validation_dir,
                                                         224,
                                                         normalize=True)

    print("train data shape: ", train_data.shape)
    print("train label shape: ", train_label.shape)
    print("validation data shape: ", validation_data.shape)
    print("validation label shape: ", validation_label.shape)

    input_size = train_data.shape[1]
    channel = train_data.shape[3]
    batch_size = 10


    mh = ModelHandler(input_size, channel)

    if model_mode == 'mymodel':
        model = mh.buildMyModel()
    elif model_mode == 'tlearn':
        model = mh.buildTlearnModel()

    model.summary()

    if do_es:
        es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    else:
        es = None

    history = model.fit(train_data,
                        train_label,
                        batch_size,
                        epochs=set_epochs,
                        validation_data=(validation_data, validation_label),
                        callbacks=es,
                        verbose=1)

    if do_es:
        child_log_dir = os.path.join(log_dir, "{}_{}_{}_with_es".format(data_mode, model_mode, no))
    else:
        child_log_dir = os.path.join(log_dir, "{}_{}_{}".format(data_mode, model_mode, no))
    os.makedirs(child_log_dir, exist_ok=True)

    # save model & weights
    model_file = os.path.join(child_log_dir, "{}_{}_{}_model.h5".format(data_mode, model_mode, no))
    model.save(model_file)

    # save history
    history_file = os.path.join(child_log_dir, "{}_{}_{}_history.pkl".format(data_mode, model_mode, no))
    with open(history_file, 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DA 実験 100例 学習プログラム")

    parser.add_argument("--earlystopping", "-es", action="store_true",
                        help="学習時に EarlyStopping を ON にする.")

    args = parser.parse_args()

    data_mode_list = ['native', 'auged']
    model_mode_list = ['mymodel', 'tlearn']

    main(data_mode_list[0],
         model_mode_list[0],
         no=0,
         do_es=args.earlystopping)


