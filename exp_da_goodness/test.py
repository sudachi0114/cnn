

import os, sys, datetime
sys.path.append(os.pardir)
now = datetime.datetime.now()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from keras.models import load_model

from utils.img_utils import inputDataCreator


def main():
    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)

    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    test_dir = os.path.join(data_dir, "test")
    print("test dir is in ... ", test_dir)

    test_data, test_label = inputDataCreator(test_dir, 224, normalize=True)

    print("test data's shape: ", test_data.shape)
    print("test label's shape: ", test_label.shape)
    print("test label: \n", test_label)


    # get model file -----
    #log_dir = os.path.join(cwd, "log")
    log_dir = os.path.join(cwd, "log_with_es")
    child_log_list = os.listdir(log_dir)

    print("\nfind logs below -----")
    for i, child in enumerate(child_log_list):
        print(i, " | ", child)

    print("\nPlease chose one child_log by index ...")
    selected_child_log_idx = input(">>> ")

    selected_child_log_dir = child_log_list[int(selected_child_log_idx)]
    child_log_dir = os.path.join(log_dir, selected_child_log_dir)

    print("\nuse log at ", child_log_dir, "\n")
    #print("this directory contain : ", os.listdir(child_log_dir))  # log list [history.pkl, model&weights.h5, log]


    child_log_list = os.listdir(child_log_dir)

    for f in child_log_list:
        if "model.h5" in f:
            model_file = os.path.join(child_log_dir, f)
    print("Use saved model : ", model_file)


    model = load_model(model_file, compile=True)

    model.summary()


    # prediction -----
    pred_result = model.predict(test_data, verbose=1)


    # class 0 -> cat / class -> dog 変換
    labels_class = []
    for i in range(len(test_label)):
        if test_label[i] == 0:
            labels_class.append('cat')
        elif test_label[i] == 1:
            labels_class.append('dog')

    # 予測結果を表に起こす
    #pred = pd.DataFrame(pred_result, columns=['dog'])
    pred = pd.DataFrame(pred_result, columns=['cat'])
    #pred['cat'] = 1.0 - pred['dog']
    pred['dog'] = 1.0 - pred['cat']
    pred['class'] = pred.idxmax(axis=1)
    pred['label'] = labels_class
    pred['collect'] = (pred['class'] == pred['label'])

    confuse = pred[pred['collect'] == False].index.tolist()
    collect = pred[pred['collect'] == True].index.tolist()

    print(pred)

    print("\nwrong recognized indeices are ", confuse)
    print("    wrong recognized amount is ", len(confuse))
    print("\ncollect recognized indeices are ", collect)
    print("    collect recognized amount is ", len(collect))
    print("\nwrong rate : ", 100*len(confuse)/len(test_label), "%")


    print("\ncheck secence ...")

    score = model.evaluate(test_data, test_label, verbose=1)
    print("test accuracy: ", score[1])
    print("test wrong rate must be (1-accuracy): ", 1.0-score[1])



    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.01, top=0.95)
    plt.subplots_adjust(hspace=0.5)
    #plt.title("Confusion picures #={}".format(len(confuse)))

    n_row = 8
    n_col = 8
    
    for i in range(2):
        if i == 0:
            j = 0
            for idx in confuse:
                plt.subplot(n_row, n_row, 1+j)
                plt.imshow(test_data[idx])
                plt.axis(False)
                plt.title("[{0}] p:{1}".format(idx, pred['class'][idx]))
                j+=1
        else:
            mod = j%n_row  # 最後の行のマスが余っている個数
            # ちょうどまであといくつ? <= n_col - あまり
            nl = j + (n_col-mod)  # newline
            for k, idx in enumerate(collect):
                plt.subplot(n_row, n_col, 1+nl+k)
                plt.imshow(test_data[idx])
                plt.axis(False)
                plt.title("[{0}] p:{1}".format(idx, pred['class'][idx]))

    img_file_place = os.path.join(child_log_dir, "{0}_AllPics_{1:%y%m%d}_{2:%H%M}.png".format(selected_child_log_dir, now, now))
    plt.savefig(img_file_place)

    print("\nexport pictures in: ", child_log_dir, "\n")




if __name__ == '__main__':

    main()
