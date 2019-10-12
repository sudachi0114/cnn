
# 学習用の画像を pickle file にして保存しておくプログラム
#   train/validation/test data を dict にまとめて保存する.


import os, pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class im2pkl():

    def __init__(self):

        # base directory define -----
        self.dirs = {}
        self.dirs['current'] = os.path.abspath(__file__)
        self.dirs['cwd'], file_base = os.path.split(self.dirs['current'])
        self.dirs['file_name'], _ = os.path.splitext(file_base)
        self.dirs['cnn_dir'] = os.path.dirname(self.dirs['cwd'])

        # data directory define -----
        self.dataset_name = 'dogs_vs_cats'
        data_size = ['smaller','full','mid300']
        self.selected_size = str(data_size[0])
        
        self.dirs['data_dir'] = os.path.join(self.dirs['cnn_dir'], '{0}_{1}'.format(self.dataset_name, self.selected_size))
        
        self.data_purpose_list = ['train', 'validation', 'test']
        self.dirs['target_dir'] = os.path.join(self.dirs['data_dir'], 'train')  # train で初期化        
       
        # define hyperparameters -----
        self.INPUT_SIZE = 224
        self.BATCH_SIZE = 10
        self.DO_SHUFFLE = False
        self.CLASS_MODE = 'binary'

    def DataGenerator(self):

        TARGET_SIZE = (self.INPUT_SIZE, self.INPUT_SIZE)

        print("Started data generator ...")
        data_gen = ImageDataGenerator()
        data_generator = data_gen.flow_from_directory(self.dirs['target_dir'],
                                                      target_size=TARGET_SIZE,
                                                      batch_size=self.BATCH_SIZE,
                                                      shuffle=self.DO_SHUFFLE,
                                                      class_mode=self.CLASS_MODE)

        print("-> Done.")
        return data_generator

    def stack(self):

        data_generator = self.DataGenerator()

        iter_num = data_generator.n//self.BATCH_SIZE

        print("Start data stacking ...")
        for i in range(iter_num):
            tmp_x, tmp_y = next(data_generator)
            if i == 0:
                x = tmp_x
                y = tmp_y
            else:
                x = np.vstack((x, tmp_x))
                y = np.hstack((y, tmp_y))

        print("-> Done.")
        return x, y

    def save2pkl(self):

        data_dict = {}
        
        for purpose in self.data_purpose_list:  # ['train', 'validation', 'test']
            self.dirs['target_dir'] = os.path.join(self.dirs['data_dir'], purpose)
            print("\nProcess {", purpose, "} data...")

            data, label = self.stack()
            data_dict['{}_data'.format(purpose)] = data
            data_dict['{}_label'.format(purpose)] = label
            print("-> Finish save", purpose, "data into data dictionary.")

        save_file = os.path.join(self.dirs['data_dir'], "data_dict.pkl")
        with open(save_file, 'wb') as p:
            pickle.dump(data_dict, p)

if __name__ == '__main__':

    changer = im2pkl()
    changer.save2pkl()

        
