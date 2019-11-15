
import os, sys, argparse, pickle, csv
sys.path.append(os.pardir)

import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)
from keras.callbacks import EarlyStopping

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator



cwd = os.getcwd()



def main(data_mode, model_mode, no, set_epochs=60, do_es=False):

    base_dir = os.path.join(cwd, "experiment_{}".format(no))

    if data_mode == 'native':
        train_dir = os.path.join(cwd, "experiment_{}".format(no), "train")
    elif data_mode == 'auged':
        train_dir = os.path.join(cwd, "concat_experiment_{}".format(no))
        set_epochs = int( set_epochs/2 )

    validation_dir = os.path.join(base_dir, "validation")
    test_dir = os.path.join(base_dir, "test")


    train_data, train_label = inputDataCreator(train_dir,
                                               224,
                                               normalize=True,
                                               one_hot=True)

    validation_data, validation_label = inputDataCreator(validation_dir,
                                                         224,
                                                         normalize=True,
                                                         one_hot=True)

    test_data, test_label = inputDataCreator(test_dir,
                                             224,
                                             normalize=True,
                                             one_hot=True)

    print("train data shape: ", train_data.shape)
    print("train label shape: ", train_label.shape)
    print("validation data shape: ", validation_data.shape)
    print("validation label shape: ", validation_label.shape)

    input_size = train_data.shape[1]
    channel = train_data.shape[3]
    batch_size = 10
    print("set epochs: ", set_epochs)


    mh = ModelHandler(input_size, channel)

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
                        batch_size,
                        epochs=set_epochs,
                        validation_data=(validation_data, validation_label),
                        callbacks=es,
                        verbose=1)

    if do_es:
        log_dir = os.path.join(cwd, "log_with_es")
    else:
        log_dir = os.path.join(cwd, "log")
    os.makedirs(log_dir, exist_ok=True)

    child_log_dir = os.path.join(log_dir, "{}_{}_{}".format(data_mode, model_mode, no))
    os.makedirs(child_log_dir, exist_ok=True)

    # save model & weights
    model_file = os.path.join(child_log_dir, "{}_{}_{}_model.h5".format(data_mode, model_mode, no))
    model.save(model_file)

    # save history
    history_file = os.path.join(child_log_dir, "{}_{}_{}_history.pkl".format(data_mode, model_mode, no))
    with open(history_file, 'wb') as p:
        pickle.dump(history.history, p)

    print("\nexport logs in ", child_log_dir)

    score = model.evaluate(test_data,
                           test_label,
                           batch_size,
                           verbose=1)

    return score


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DA 実験 100例 学習プログラム")

    parser.add_argument("--earlystopping", "-es", action="store_true",
                        help="学習時に EarlyStopping を ON にする.")

    args = parser.parse_args()

    data_mode_list = ['native', 'auged']
    model_mode_list = ['mymodel', 'tlearn']

    test_acc_list = []
    test_loss_list = []

    csv_dir = os.path.join(cwd, "csv")
    os.makedirs(csv_dir, exist_ok=True)


    for i in range(5):
        for data_mode in data_mode_list:
            for model_mode in model_mode_list:
                print("========== No:{} | data:{} | model:{} ==========".format(i, data_mode, model_mode))
                score = main(data_mode,
                             model_mode,
                             no=i,
                             #set_epochs=5,
                             do_es=args.earlystopping)

                test_loss_list.append(score[0])
                test_acc_list.append(score[1])

                to_csv = [test_loss_list, test_acc_list]
                with open(os.path.join(csv_dir, "{}_{}_{}_test_score.csv".format(i, data_mode, model_mode)), "w") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerows(to_csv)

                    print("\nexport {} {} {} test score as CSV.".format(data_mode, model_mode, i))


