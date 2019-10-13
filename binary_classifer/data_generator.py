
# 訓練データを generate するプログラム

import os, pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator():

    def __init__(self):

        # attribute -----
        self.DO_RESCALE = True
        self.INPUT_SIZE = 224
        self.BATCH_SIZE = 10
        self.CLASS_MODE = 'binary'

    # DA などを行わない基本的な Generator
    def Generator(self, target_dir):

        print("\nStart DataGenerator ...")

        if self.DO_RESCALE:
            datagen = ImageDataGenerator(rescale=1.0/255.0)
        else:
            datagen = ImageDataGenerator()

        TARGET_SIZE = (self.INPUT_SIZE, self.INPUT_SIZE)

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

    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")


    # check DataGenerator -----
    train_dir = os.path.join(data_dir, "train")
    
    train_generator = DataGenerator().DataGenerator(train_dir)

    data_checker, label_checker = next(train_generator)

    print("--*--"*5)

    # check npzLoader -----
    target_file = os.path.join(data_dir, "train.npz")

    data_checker, label_checker = DataGenerator().npzLoader(target_file)
    
    print("data_checker's shape: ", data_checker.shape)
    print("labal_checker's shape: ", label_checker.shape)

    print("--*--"*5)

    # check pklLoader -----
    target_file = os.path.join(data_dir, "data_dict.pkl")

    data_dict = DataGenerator().pklLoader(target_file)

    print("data_dict keys: ", data_dict.keys())
