
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

class DaHandler:

    def __init__(self):

        self.dirs = {}
        self.dirs['cwd'] = os.getcwd()
        self.dirs['cnn_dir'] = os.path.dirname(self.dirs['cwd'])
        self.dirs['data_dir'] = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_smaller")

        self.train_file = os.path.join(self.dirs['data_dir'], "train.npz")
        self.validation_file = os.path.join(self.dirs['data_dir'], "validation.npz")
        self.test_file = os.path.join(self.dirs['data_dir'], "test.npz")

        # attributes -----
        self.BATCH_SIZE = 10
        self.DO_SHUFFLE = True


    def validationData(self):
        npz = np.load(self.validation_file)
        validation_data, validation_label = npz['data'], npz['label']

        return validation_data, validation_label

    def testData(self):
        npz = np.load(self.test_file)
        test_data, test_label = npz['data'], npz['label']

        return test_data, test_label

    def trainData(self):
        npz = np.load(self.train_file)
        train_data, train_label = npz['data'], npz['label']

        return train_data, train_label

    def data_augment_keras(self, mode='native'):

        if mode == 'native':
            keras_da = ImageDataGenerator(rescale=1.0/255.0)
        elif mode == 'rotation':  # 個人的には rotation は DA の中でも効果を発揮してくれると思っている..
            keras_da = ImageDataGenerator(rescale=1.0/255.0,
                                          rotation_range=90)  # 回転 (max 90度まで)
        elif mode == 'hflip':
            keras_da = ImageDataGenerator(rescale=1.0/255.0,
                                          horizontal_flip=True)  # 左右反転
        elif mode == 'width_shift':
            keras_da = ImageDataGenerator(rescale=1.0/255.0,
                                          width_shift_range=0.125)  # 1/8 平行移動(左右)
        elif mode == 'height_shift':
            keras_da = ImageDataGenerator(rescale=1.0/255.0,
                                          height_shift_range=0.125)  # 1/8 平行移動(上下)
        elif mode == 'zoom':
            keras_da = ImageDataGenerator(rescale=1.0/255.0,
                                          zoom_range=0.2)  # (0.8 ~ 1.2 の間で) 拡大/縮小
        #elif mode == 'fwize_center':
        #    keras_da = ImageDataGenerator(rescale=1.0/255.0,
        #                                  featurewise_center=True)  # 平均を0に正規化(入力wiseに)
        elif mode == 'swize_center':
            keras_da = ImageDataGenerator(rescale=1.0/255.0,
                                          samplewise_center=True)  # 平均を0に正規化(画像1枚wiseに)
        #elif mode == 'fwize_std_normalize':
        #    keras_da = ImageDataGenerator(rescale=1.0/255.0,
        #                                  featurewise_std_normalization=True)  # 標準偏差正規化(入力wiseに)
        elif mode == 'swize_std_normalize':
            keras_da = ImageDataGenerator(rescale=1.0/255.0,
                                          samplewise_std_normalization=True)  # 標準偏差正規化(画像1枚wiseに)
        elif mode == 'vflip':
            keras_da = ImageDataGenerator(rescale=1.0/255.0,
                                          vertical_flip=True)  # 上下反転
        elif mode == 'standard':
            keras_da = ImageDataGenerator(rescale=1.0/255.0,
                                           horizontal_flip=True,
                                           width_shift_range=0.125,
                                           height_shift_range=0.125)
        else:
            raise ValueError("正しくないモードが選択されています。")

        train_data, train_label = self.trainData()

        data_generator = keras_da.flow(train_data,
                                       train_label,
                                       batch_size=self.BATCH_SIZE,
                                       shuffle=self.DO_SHUFFLE)

        return data_generator

    def display(self):

        for n_confirm in range(3):  # 三回出力して確認
            self.DO_SHUFFLE = False
            data_generator = self.data_augment_keras(mode='fwize_center')

            data_checker, label_checker = next(data_generator)

            plt.figure(figsize=(12, 6))

            for i in range(len(label_checker)):
                plt.subplot(2, 5, i+1)
                plt.imshow(data_checker[i])
                plt.title("l: [{}]".format(label_checker[i]))
                plt.axis(False)

            plt.show()




if __name__ == '__main__':

    da_handler = DaHandler()
    validation_data, validation_label = da_handler.validationData()

    print("validation_data's shape: ", validation_data.shape)
    print("validation_label's shape: ", validation_label.shape)

    test_data, test_label = da_handler.testData()

    print("test_data's shape: ", test_data.shape)
    print("test_label's shape: ", test_label.shape)

    train_data, train_label = da_handler.trainData()

    print("train_data's shape: ", train_data.shape)
    print("train_label's shape: ", train_label.shape)

    data_generator = da_handler.data_augment_keras()
    data_checker, label_checker = next(data_generator)

    print("data_checker's shape: ", data_checker.shape)
    print("label_checker's shape: ", label_checker.shape)

    da_handler.display()
