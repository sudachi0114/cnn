
import os, sys
sys.path.append(os.pardir)

import time, datetime, gc
import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
sess = tf.Session(config=config)
K.set_session(sess)

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator, dataSplit


def main():

    cwd = os.getcwd()

    log_dir = os.path.join(cwd, "log")
    os.makedirs(log_dir, exist_ok=True)

    base_dir = os.path.dirname(cwd)
    data_dir = os.path.join(base_dir, "dogs_vs_cats_smaller", "train")


    print("\ncreate train data")
    total_data, total_label = inputDataCreator(data_dir,
                                               224,
                                               normalize=True,
                                               one_hot=True)

    train_data, train_label, validation_data, validation_label, test_data, test_label = dataSplit(total_data, total_label)

    print(train_data.shape)
    print(validation_data.shape)
    print(test_data.shape)
    print(test_label)


    mh = ModelHandler(224, 3)

    model = mh.buildTlearnModel(base='mnv1')

    model.summary()

    print("\ntraining sequence started...")
    start = time.time() 
    history = model.fit(train_data,
                        train_label,
                        batch_size=10,
                        epochs=30,
                        validation_data=(validation_data, validation_label),
                        verbose=1)
    elapsed_time = time.time() - start
    print("  total elapsed time: {} [sec]".format(elapsed_time))
    
    accs = history.history['accuracy']
    losses = history.history['loss']
    val_accs = history.history['val_accuracy']
    val_losses = history.history['val_loss']
    print("last val_acc: ", val_accs[len(val_accs)-1])

    # make log_dirctory ----------
    now = datetime.datetime.now()
    child_log_dir = os.path.join(log_dir, "{0:%Y%m%d}".format(now))
    os.makedirs(child_log_dir, exist_ok=True)

    # save model & weights
    model_file = os.path.join(child_log_dir, "model.h5")
    model.save(model_file)
    print("\nexport model in ", child_log_dir)


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
                              batch_size=10,
                              verbose=1)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])


    # save history
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

    df_result = pd.DataFrame(save_dict.values(), index=save_dict.keys())

    csv_file = os.path.join( child_log_dir, "result.csv" )
    df_result.to_csv(csv_file)
    print("\nexport history in ", csv_file)



if __name__ == '__main__':
    main()
