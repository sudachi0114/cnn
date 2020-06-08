
import os, sys, argparse, pickle, csv, time
sys.path.append(os.pardir)

import pandas as pd

import tensorflow as tf
import keras
import gc  # gabage collection
"""
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.1
sess = tf.Session(config=config)
K.set_session(sess)
"""

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from keras.callbacks import EarlyStopping

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator




cwd = os.getcwd()



def main(N, LEARN_PATH, DATA_MODE, EPOCHS=60, FINE_TUNE_AT=81):


    sample_dir = os.path.join(LEARN_PATH, "sample_{}".format(N))

    use_da_data = False
    if use_da_data:
        train_dir = os.path.join(sample_dir, "train_with_aug")
    else:
        train_dir = os.path.join(sample_dir, "train")
    validation_dir = os.path.join(sample_dir, "validation")
    test_dir = os.path.join(sample_dir, "test")

    print("train_dir: ", train_dir)
    print("validation_dir: ", validation_dir)
    print("test_dir: ", test_dir)


    # data load ----------
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

    print("\ntrain data shape: ", train_data.shape)
    print("train label shape: ", train_label.shape)
    print("\nvalidation data shape: ", validation_data.shape)
    print("validation label shape: ", validation_label.shape)

    input_size = train_data.shape[1]
    channel = train_data.shape[3]
    batch_size = 10
    print("set epochs: ", EPOCHS)


    # build model ----------
    mh = ModelHandler(input_size, channel)

    # あとで重みの解凍をできるように base_model を定義
    base_model = mh.buildMnv1Base()
    base_model.trainable=False

    model = mh.addChead(base_model)

    model.summary()

    """
    es = EarlyStopping(monitor='val_loss',
                       patience=5,
                       verbose=1,
                       mode='auto',
                       restore)
    """
    # early stopping
    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       restore_best_weights=True)


    print("\ntraining sequence start .....")
    start = time.time()

    # 準備体操 -----
    print("\nwarm up sequence .....")
    model.summary()
    _history = model.fit(train_data,
                         train_label,
                         batch_size,
                         epochs=EPOCHS,
                         validation_data=(validation_data, validation_label),
                         callbacks=[es],
                         verbose=2)

    mh.setFineTune(base_model, model, FINE_TUNE_AT)
    model.summary()

    history = model.fit(train_data,
                        train_label,
                        batch_size,
                        epochs=EPOCHS,
                        validation_data=(validation_data, validation_label),
                        callbacks=[es],
                        verbose=2)


    elapsed_time = time.time() - start
    print( "elapsed time (for train): {} [sec]".format(time.time() - start) )

    accs = history.history['accuracy']
    losses = history.history['loss']
    val_accs = history.history['val_accuracy']
    val_losses = history.history['val_loss']


    print("\npredict sequence...")
    pred = model.predict(test_data,
                         batch_size=10,
                         verbose=1)

    label_name_list = []
    for i in range(len(test_label)):
        if test_label[i][0] == 1:
            label_name_list.append('cat')
        elif test_label[i][1] == 1:
            label_name_list.append('dog')

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
                              batch_size=10,
                              verbose=1)

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
    save_dict['elapsed_time'] = elapsed_time

    print(save_dict)

    # 重そうなものは undefine してみる
    #del train_data, train_label, validation_data, validation_label, test_data, test_label
    del model
    del _history, history
    #del pred

    keras.backend.clear_session()
    gc.collect()

    return save_dict


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="DA 実験 100例 学習プログラム")
    # args = parser.parse_args()

    data_mode_list = ['native', 'auged']

    select_data = 'auged'
    print("\nuse data:{}".format(select_data))


    learn_path = "/home/sudachi/cnn/datasets/mulSample/1000_721_Mul"
    
    N = 5
    for i in range(N):
        print("\ndata no. {} -------------------------------".format(i))
        result_dict = main(N=i,
                           LEARN_PATH=learn_path,
                           DATA_MODE=select_data,
                           EPOCHS=60,
                           FINE_TUNE_AT=81)
        if i == 0:
            df_result = pd.DataFrame(result_dict.values(), index=result_dict.keys())
            """
                ['last_loss',
                 'last_acc',
                 'last_val_loss',
                 'last_val_acc',
                 'n_confuse',
                 'eval_loss',
                 'eval_acc',
                 'elapsed_time']
            """

        else:
            series = pd.Series(result_dict)
            df_result[i] = series
        print(df_result)

    csv_file = "./{}_{}_result.csv".format(select_data, select_model)
    df_result.to_csv(csv_file)

    print("\nexport {}  as CSV.".format(csv_file))

