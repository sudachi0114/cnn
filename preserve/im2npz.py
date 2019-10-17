
# 学習用の画像を numpy 配列にして保存しておくプログラム
#   # 注意!!
#   #   npz で保存すると、入力画像の解像度 (サンプリングサイズ) を静的に固定することになるので
#   #   train (validation) / test の model の入力層はデータの解像度にあわせるようにする。

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class im2npz():

    def __init__(self):

        # define directory dependences -----
        self.dirs = {}
        self.dirs['current'] = os.path.abspath(__file__)
        self.dirs['cwd'], self.file_name = os.path.split(self.dirs['current'])
        self.dirs['cnn_dir'] = os.path.dirname(self.dirs['cwd'])

        self.dataset_name = 'dogs_vs_cats'
        self.data_size = 'smaller'  # ['smaller','full','mid300']
        self.data_purpose_list = ['train', 'validation', 'test']
        self.data_purpose = 'train'  # ['train', 'validation', 'test']
        
        self.dirs['data_dir'] = os.path.join(self.dirs['cnn_dir'], '{0}_{1}'.format(self.dataset_name, self.data_size))

        # define hyper parameters -----
        self.INPUT_SIZE = 224
        self.CHANNEL = 3
        self.BATCH_SIZE = 10
        self.CLASS_MODE = 'binary'
        self.DO_SHUFFLE = False

    def DataGenerator(self):

        # set target dir -----
        self.dirs['target_dir'] = os.path.join(self.dirs['data_dir'], self.data_purpose)
        print("Convert {", self.dirs['target_dir'], "} data to npz file...")

        print("Start data generation ...")
        TARGET_SIZE = (self.INPUT_SIZE, self.INPUT_SIZE)

        datagen = ImageDataGenerator()
        data_generator = datagen.flow_from_directory(self.dirs['target_dir'],
                                                     target_size=TARGET_SIZE,
                                                     batch_size=self.BATCH_SIZE,
                                                     shuffle=self.DO_SHUFFLE,
                                                     class_mode=self.CLASS_MODE)
        print("-> Done.")
        return data_generator

    def stack(self):

        data_generator = self.DataGenerator()
        #data_checker, label_checker = next(data_generator)

        iter_num = data_generator.n//self.BATCH_SIZE

        print("Start stacking datas ...")

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

    def save2npz(self):

        print("\nSave datas into npz file ...")

        for purpose in self.data_purpose_list:

            print("\nProcess ", self.data_purpose, "data...")
            self.data_purpose = purpose

            x, y = self.stack()
        
            save_file = os.path.join(self.dirs['data_dir'], '{}.npz'.format(self.data_purpose))
            np.savez(save_file, data=x, label=y)

            print("Collectory Saved!: ", save_file)

if __name__ == '__main__':

    changer = im2npz()
    changer.save2npz()

