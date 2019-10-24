
# 誤認識した画像をプロットして考察する:


import os, sys, datetime
sys.path.append(os.pardir)
now = datetime.datetime.now()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from data_handler import DataHandler


class errorAnalysis:

    def __init__(self):

        self.dirs = {}
        self.dirs['cwd'] = os.getcwd()
        self.dirs['cnn_dir'] = os.path.dirname(self.dirs['cwd'])

        # get test data -----
        self.dirs['data_dir'] = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_smaller")
        #test_dir = os.path.join(data_dir, "test")
        #print("test dir is in ... ", test_dir)

        self.target_file = os.path.join(self.dirs['data_dir'], "test.npz")

        # get model file -----
        self.dirs['log_dir'] = os.path.join(self.dirs['cwd'], "log")
        child_log_list = os.listdir(self.dirs['log_dir'])

        print("\nfind logs below -----")
        for i, child in enumerate(child_log_list):
            print(i, " | ", child)

        print("\nPlease chose one child_log by index ...")
        selected_child_log_idx = input(">>> ")

        self.selected_child_log_dir = child_log_list[int(selected_child_log_idx)]
        self.dirs['child_log_dir'] = os.path.join(self.dirs['log_dir'], self.selected_child_log_dir)

        print("\nuse log at ", self.dirs['child_log_dir'], "\n")
        #print("this directory contain : ", os.listdir(child_log_dir))  # log list [history.pkl, model&weights.h5, log]


        child_log_list = os.listdir(self.dirs['child_log_dir'])

        for f in child_log_list:
            if "model" in f:
                self.model_file = os.path.join(self.dirs['child_log_dir'], f)
        print("Use saved model : ", self.model_file)

    # old: def testDataGenerator(test_dir, input_size, batch_size=10):
    def dataLoader(self):

        data_handler = DataHandler()
        npz = data_handler.npzLoader(self.target_file)

        data, label = npz[0], npz[1]
        data /= 255.0

        return data, label


    def reloadModel(self):

        model = load_model(self.model_file, compile=True)

        return model


    def predictor(self, model, data, label):

        pred_result = model.predict(data, verbose=1)

        return pred_result  # クラス 1 の予測値(確率)のnp配列


    def evaluator(self, model, data, label):

        # wrong rate の check 用
        print("\ncheck secence ...")

        score = model.evaluate(data, label, verbose=1)

        print("test accuracy: ", score[1])
        print("test wrong rate must be (1-accuracy): ", 1.0-score[1])



    def displayConfuse(self, data, label, pred_result):

        # class 0 -> cat / class -> dog 変換
        labels_class = []
        for i in range(len(label)):
            if label[i] == 0:
                labels_class.append('cat')
            elif label[i] == 1:
                labels_class.append('dog')

        # 予測結果を表に起こす
        pred = pd.DataFrame(pred_result, columns=['dog'])
        pred['cat'] = 1.0 - pred['dog']
        pred['class'] = pred.idxmax(axis=1)
        pred['label'] = labels_class
        pred['collect'] = (pred['class'] == pred['label'])
        print(pred)


        confuse = pred[pred['collect'] == False].index.tolist()
        print("\nwrong recognized indeices are ", confuse)
        print("wrong recognized amount is ", len(confuse))
        print("wrong rate : ", 100*len(confuse)/len(label), "%")

        # display wrong classified photo -----
        plt.figure(figsize=(7, 6))
        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        #plt.title("Confusion picures #={}".format(len(confuse)))

        for i, idx in enumerate(confuse):
            plt.subplot(7, 7, 1+i)
            plt.imshow(data[idx])
            plt.axis(False)
            plt.title("[{0}] p:{1}".format(idx, pred['class'][idx]))
        img_file_place = os.path.join(self.dirs['child_log_dir'], "{0}_confusePics_{1:%y%m%d}_{2:%H%M}.png".format(self.selected_child_log_dir, now, now))
        plt.savefig(img_file_place)


    def displayCollect(self, data, label, pred_result):

        # class 0 -> cat / class -> dog 変換
        labels_class = []
        for i in range(len(label)):
            if label[i] == 0:
                labels_class.append('cat')
            elif label[i] == 1:
                labels_class.append('dog')

        # 予測結果を表に起こす
        pred = pd.DataFrame(pred_result, columns=['dog'])
        pred['cat'] = 1.0 - pred['dog']
        pred['class'] = pred.idxmax(axis=1)
        pred['label'] = labels_class
        pred['collect'] = (pred['class'] == pred['label'])

        collect = pred[pred['collect'] == True].index.tolist()

        # display collect classified photo -----
        plt.figure(figsize=(7, 6))
        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        #plt.title("Confusion picures #={}".format(len(confuse)))

        for i, idx in enumerate(collect):
            plt.subplot(7, 7, 1+i)
            plt.imshow(data[idx])
            plt.axis(False)
            plt.title("[{0}] p:{1}".format(idx, pred['class'][idx]))
        img_file_place = os.path.join(self.dirs['child_log_dir'], "{0}_collectPics_{1:%y%m%d}_{2:%H%M}.png".format(self.selected_child_log_dir, now, now))
        plt.savefig(img_file_place)


    def displayAll(self, data, label, pred_result):

        labels_class = []
        for i in range(len(label)):
            if label[i] == 0:
                labels_class.append('cat')
            elif label[i] == 1:
                labels_class.append('dog')

        # 予測結果を表に起こす
        pred = pd.DataFrame(pred_result, columns=['dog'])
        pred['cat'] = 1.0 - pred['dog']
        pred['class'] = pred.idxmax(axis=1)
        pred['label'] = labels_class
        pred['collect'] = (pred['class'] == pred['label'])
        print(pred)

        confuse = pred[pred['collect'] == False].index.tolist()
        collect = pred[pred['collect'] == True].index.tolist()
        print("\nwrong recognized indeices are ", confuse)
        print("wrong recognized amount is ", len(confuse))
        print("\ncollect recognized indeices are ", collect)
        print("collect recognized amount is ", len(collect))
        print("wrong rate : ", 100*len(confuse)/len(label), "%")


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
                    plt.imshow(data[idx])
                    plt.axis(False)
                    plt.title("[{0}] p:{1}".format(idx, pred['class'][idx]))
                    j+=1
            else:
                mod = j%n_row  # 最後の行のマスが余っている個数
                # ちょうどまであといくつ? <= n_col - あまり
                nl = j + (n_col-mod)  # newline
                for k, idx in enumerate(collect):
                    plt.subplot(n_row, n_col, 1+nl+k)
                    plt.imshow(data[idx])
                    plt.axis(False)
                    plt.title("[{0}] p:{1}".format(idx, pred['class'][idx]))

        img_file_place = os.path.join(self.dirs['child_log_dir'], "{0}_AllPics_{1:%y%m%d}_{2:%H%M}.png".format(self.selected_child_log_dir, now, now))
        plt.savefig(img_file_place)



    def do_whole(self):

        model = self.reloadModel()
        model.summary()

        test_data, test_label = self.dataLoader()

        pred_result = self.predictor(model, test_data, test_label)
        #print("pred_result : ", pred_result)
        #print(labels)

        self.displayConfuse(test_data, test_label, pred_result)
        self.displayCollect(test_data, test_label, pred_result)
        self.displayAll(test_data, test_label, pred_result)

        # wrong rate check
        self.evaluator(model, test_data, test_label)



if __name__ == '__main__':

    task = errorAnalysis()
    task.do_whole()
