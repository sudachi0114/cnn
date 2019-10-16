
# 訓練データを用意するプログラム

import os, pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class DataHandler:

    def __init__(self):

        # directory ------
        self.dirs = {}
        self.dirs['cwd'] = os.getcwd()
        self.dirs['cnn_dir'] = os.path.dirname(self.dirs['cwd'])

        dataset_list = ['dogs_vs_cats']
        dataset_size_list = ['smaller', 'mid300', 'full']

        self.dataset = dataset_list[0]  # 冗長な書き方 (FIXME:)
        self.dataset_size = dataset_size_list[0]

        selected_dataset = '{}_{}'.format(self.dataset, self.dataset_size)
        
        self.dirs['data_dir'] = os.path.join(self.dirs['cnn_dir'], selected_dataset)

        data_purpose_list = ['train', 'validation', 'test']

        self.data_purpose = ''

        # attribute -----
        self.DO_RESCALE = True
        self.INPUT_SIZE = 224
        self.BATCH_SIZE = 10
        self.CLASS_MODE = 'binary'


    # DA などを行わない基本的な Generator
    def dataGenerator(self, target_dir=''):

        print("\nStart DataGenerator ...")

        if self.DO_RESCALE:
            datagen = ImageDataGenerator(rescale=1.0/255.0)
        else:
            datagen = ImageDataGenerator()

        TARGET_SIZE = (self.INPUT_SIZE, self.INPUT_SIZE)

        
        if target_dir == '':
            target_dir = os.path.join(self.dirs['data_dir'], self.data_purpose)
        else:
            target_dir = self.target_dir

        data_generator = datagen.flow_from_directory(target_dir,
                                                     target_size=TARGET_SIZE,
                                                     batch_size=self.BATCH_SIZE,
                                                     class_mode=self.CLASS_MODE)

        print("-> Generation has completed.")

        return data_generator

    
    def npzLoader(self, target_file):

        print("\nStart Loading npz file ...")

        npz = np.load(target_file)
        data, label = npz['data'], npz['label']

        print("-> Collectly load data.")

        return data, label

    
    
    def pklLoader(self, target_file):

        print("\nStart Loading pickle file ...")
                
        with open(target_file, 'rb') as p:
            data_dict = pickle.load(p)

        print("-> Collectly load data.")

        return data_dict




if __name__ == '__main__':

    cnn_dir = os.getcwd()
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")

    data_handler = DataHandler()

    # check DataGenerator -----
    train_dir = os.path.join(data_dir, "train")
    
    train_generator = data_handler.dataGenerator(train_dir)

    data_checker, label_checker = next(train_generator)

    print("--*--"*5)

    # check npzLoader -----
    target_file = os.path.join(data_dir, "train.npz")

    data_checker, label_checker = data_handler.npzLoader(target_file)
    
    print("data_checker's shape: ", data_checker.shape)
    print("labal_checker's shape: ", label_checker.shape)

    print("--*--"*5)

    # check pklLoader -----
    target_file = os.path.join(data_dir, "data_dict.pkl")

    data_dict = data_handler.pklLoader(target_file)

    print("data_dict keys: ", data_dict.keys())
