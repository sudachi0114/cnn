
# verify Data augmentation has power of opposition to noise test data
import os, sys
sys.path.append(os.pardir)

import time, datetime, gc
import numpy as np
import pandas as pd


import tensorflow as tf
import keras
from keras import backend as K
print("TensorFlow version is ", tf.__version__)
print("Keras version is ", keras.__version__)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
sess = tf.Session(config=config)
K.set_session(sess)

from keras.preprocessing.image import ImageDataGenerator

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator, dataSplit

from sklearn.metrics import confusion_matrix


# define -----

def main():

    cwd = os.getcwd()
    sub_prj = os.path.dirname(cwd)
    sub_prj_root = os.path.dirname(sub_prj)
    prj_root = os.path.dirname(sub_prj_root)


    data_dir = os.path.join(prj_root, "datasets")

    # data_src = os.path.join(data_dir, "small_721")
    data_src = os.path.join(data_dir, "cdev_origin")
    print("\ndata source: ", data_src)

    """
    use_da_data = False
    increase_val = False
    print( "\nmode: Use Augmented data: {} | increase validation data: {}".format(use_da_data, increase_val) )

    # First define original train_data only as train_dir
    train_dir = os.path.join(data_dir, "train")
    if (use_da_data == True) and (increase_val == False):
        # with_augmented data (no validation increase)
        train_dir = os.path.join(data_dir, "train_with_aug")
    validation_dir = os.path.join(data_dir, "val")  # original validation data

    # pair of decreaced train_data and increased validation data
    if (increase_val == True):
        train_dir = os.path.join(data_dir, "red_train")
        if (use_da_data == True):
            train_dir = os.path.join(data_dir, "red_train_with_aug")
        validation_dir = os.path.join(data_dir, "validation")
    """
    


    print("\ncreate train data")
    total_data, total_label = inputDataCreator(data_src,
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


    # confusion matrix -----
    #           Predict
    #           0  | 1
    #       --+----+-----
    #       0 | TN | FP
    # label -------+-----
    #       1 | FN | TP

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

    # 適合率 (precision):
    # precision = tp/(tp+fp)
    # print("Precision of the model is {}".format(precision))
    
    # 再現率 (recall):
    # recall = tp/(tp+fn)
    # print("Recall of the model is {}".format(recall))



    # save some result score, model & weights ----------
    now = datetime.datetime.now()
    log_dir = os.path.join(sub_prj, "outputs")
    child_log_dir = os.path.join(log_dir, "{0:%Y%m%d}".format(now))
    # os.makedirs(child_log_dir, exist_ok=True)

    save_location = os.path.join(log_dir, "models")
    save_file = os.path.join(save_location, "model.h5")
    model.save(save_file)
    print("\nmodel has saved in", save_file)


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
