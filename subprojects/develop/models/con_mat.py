
# confusion_matrix を作ってより精密な評価

import os, sys
sys.path.append(os.pardir)
import time, datetime, gc
import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.5
sess = tf.Session(config=config)
K.set_session(sess)

# from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator, dataSplit


# define -----
batch_size = 50
input_size = 224
channel = 3
target_size = (input_size, input_size)
input_shape = (input_size, input_size, channel)
set_epochs = 40


def main():

    cwd = os.getcwd()
    sub_prj = os.path.dirname(cwd)
    sub_prj_root = os.path.dirname(sub_prj)
    prj_root = os.path.dirname(sub_prj_root)

    data_dir = os.path.join(prj_root, "datasets")

    data_src = os.path.join(data_dir, "small_721")
    print("\ndata source: ", data_src)

    use_da_data = False
    if use_da_data:
        train_dir = os.path.join(data_src, "train_with_aug")
    else:
        train_dir = os.path.join(data_src, "train")
    validation_dir = os.path.join(data_src, "validation")
    test_dir = os.path.join(data_src, "test")

    print("train_dir: ", train_dir)
    print("validation_dir: ", validation_dir)
    print("test_dir: ", test_dir)


    # data load ----------
    train_data, train_label = inputDataCreator(train_dir,
                                               input_size,
                                               normalize=True,
                                               one_hot=True)
    validation_data, validation_label = inputDataCreator(validation_dir,
                                                         input_size,
                                                         normalize=True,
                                                         one_hot=True)
    test_data, test_label = inputDataCreator(test_dir,
                                             input_size,
                                             normalize=True,
                                             one_hot=True)
    """
    total_data, total_label = inputDataCreator(data_dir,
                                               224,
                                               normalize=True,
                                               one_hot=True)

    train_data, train_label, validation_data, validation_label, test_data, test_label = dataSplit(total_data, total_label)
    """

    print("train data shape (in batch): ", train_data.shape)
    print("train label shape (in batch): ", train_label.shape)
    # print("validation data shape:", validation_data.shape)
    # print("validation label shape:", validation_label.shape)
    # print("test data shape:", test_data.shape)
    # print("test label shape:", test_label.shape)


    # build model ----------
    mh = ModelHandler(input_size, channel)
    model = mh.buildMyModel()
    model.summary()


    # instance EarlyStopping -----
    es = EarlyStopping(monitor='val_loss',
                       # monitor='val_accuracy',
                       patience=5,
                       verbose=1,
                       restore_best_weights=True)


    print("\ntraining sequence start .....")
    start = time.time() 
    history = model.fit(train_data,
                        train_label,
                        batch_size=batch_size,
                        epochs=set_epochs,
                        validation_data=(validation_data, validation_label),
                        callbacks=[es],
                        verbose=1)
    elapsed_time = time.time() - start
    print( "elapsed time (for train): {} [sec]".format(elapsed_time) )
    

    # evaluate ----------
    print("\nevaluate sequence...")

    accs = history.history['accuracy']
    losses = history.history['loss']
    val_accs = history.history['val_accuracy']
    val_losses = history.history['val_loss']
    print("last val_acc: ", val_accs[len(val_accs)-1])
    
    eval_res = model.evaluate(test_data,
                              test_label,
                              batch_size=10,
                              verbose=1)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])


    # logging and detail outputs -----
    # make log_dirctory
    log_dir = os.path.join(sub_prj, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    model_log_dir = os.path.join(sub_prj, "outputs", "models")
    os.makedirs(log_dir, exist_ok=True)

    now = datetime.datetime.now()
    child_log_dir = os.path.join(log_dir, "{0:%Y%m%d}".format(now))
    os.makedirs(child_log_dir, exist_ok=True)
    child_model_log_dir = os.path.join(model_log_dir, "{0:%Y%m%d}".format(now))
    os.makedirs(child_model_log_dir, exist_ok=True)

    # save model & weights
    model_file = os.path.join(child_model_log_dir, "model.h5")
    model.save(model_file)
    print("\nexport model in ", child_model_log_dir)


    print("\npredict sequence...")
    pred = model.predict(test_data,
                         batch_size=batch_size,
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


    # confusion matrix -----
    #           Predict
    #           0  | 1
    #       --+----+-----
    #       0 | TN | FP
    # label -------+-----
    #       1 | FN | TP
    print("\nconfusion matrix")

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
    #   sklearn.metrics => precision_score() にも libaray がある。
    #       # 入力は (idx_label, idx_pred)
    #       「陽性と予測されたサンプルのうち, 正解したサンプルの割合」
    #       PPV (positive predictive value) とも呼ばれる。
    precision = tp/(tp+fp)
    print("Precision of the model is {}".format(precision))
    
    # 再現率 (recall:
    #    sklearn.metrics => recall_score() にも library がある
    #       # 入力は (idx_label, idx_pred)
    #    「実際に陽性のサンプルのうち, 正解したサンプルの割合」。
    #        sensitivity や hit rate,
    #        TPR (true positive rate, 真陽性率) などとも呼ばれる。
    #         # すごく大雑把にいえば
    #         #    class 1 だけに対して考えた正解率の様な指標だと言える
    #         #    (Negative 側がどれだけ正解/不正解かは don't care)
    #         # 逆に TN / (TN + FP) とすると
    #         #    class 0 だけに対する正解率となる。
    recall = tp/(tp+fn)
    print("Recall of the model is {}".format(recall))


    # 正解率 (accuracy) も confusion_matrix から計算できるが
    #    既に keras の method を用いて出力することができているので
    #    ここでは扱わない。
    # 実は他にも F1-score (F1-measure)
    #    マクロ平均, マイクロ平均, 加重平均
    #    と呼ばれる指標もある..



if __name__ == '__main__':
    main()
