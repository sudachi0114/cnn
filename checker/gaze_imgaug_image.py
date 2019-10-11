
# Data Argumentation の効果検証用プログラム
"""
---- keras でできる -----
回転 (90度(まで))
左右反転
平行移動(縦横 1/8)
拡大縮小(80% ~ 120% - 精度下がる??)
正規化 (2つ)
    平均(featurewise/samplewise)
    標準偏差(featurewise/samplewise)
=> da_all.py を参照

----- 別プログラム (keras +alpha) -----
明度変換
コントラスト変換
ノイズ付加 (3つ)
    Gaussian Noise
    Laplace Noise
    Poisson Noise
平滑化
鮮鋭化
色反転

# keras -> ImageDataGenerator を用いて画像を読み込み (早いのでこれを使う)
#   batch個読み込んだものを stack して np.ndarray で持つ
#   ndarray で持った画像に対して imgaug を用いて変換を行う
"""

import os
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa

import tensorflow as tf

"""
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)
"""

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# 共通
def validationDataGenerator(validation_dir, INPUT_SIZE=224, batch_size=10):

    print("\nstarting validationDataGenerator ...")

    TARGET_SIZE = (INPUT_SIZE, INPUT_SIZE)
    
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)    
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=TARGET_SIZE,
                                                                  batch_size=batch_size,
                                                                  class_mode='binary')
    return validation_generator

# これをいじっていく
def trainDataGenerator(train_dir, INPUT_SIZE=224, batch_size=10):

    print("\nstarting trainDataGenerator ...")

    TARGET_SIZE = (INPUT_SIZE, INPUT_SIZE)
    
    train_datagen = ImageDataGenerator(rescale=1.0/255.0
                                       #rotation_range=90,
                                       #horizontal_flip=True,
                                       #width_shift_range=0.125,
                                       #height_shift_range=0.125,
                                       #zoom_range=0.2,
                                       #featurewise_center=True,
                                       #featurewise_std_normalization=True
                                       )

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=TARGET_SIZE,
                                                        batch_size=batch_size,
                                                        class_mode='binary')
    return train_generator


# iterator -> 溜め込んで配列で持つので
#   訓練メソッドも fit_generator ではなく、fit を使う
#   そのため、validation data も stack する
def validation_data_stocker(validation_dir):

    x_validation, y_validation = [], []

    validation_generator = validationDataGenerator(validation_dir)

    data_checker, label_checker = next(validation_generator)
    batch_size = data_checker.shape[0]
    
    validation_iter_num = validation_generator.n//batch_size

    for i in range(validation_iter_num):
        tmp_x, tmp_y = next(validation_generator)
        if i == 0:
            x_validation = tmp_x
            y_validation = tmp_y
        else:
            x_validation = np.vstack((x_validation, tmp_x))
            y_validation = np.hstack((y_validation, tmp_y))

    return x_validation, y_validation


def train_data_stocker(train_dir):

    x_train, y_train = [], []

    train_generator = trainDataGenerator(train_dir)

    data_checker, label_checker = next(train_generator)
    batch_size = data_checker.shape[0]

    train_iter_num = train_generator.n//batch_size

    for i in range(train_iter_num):
        tmp_x, tmp_y = next(train_generator)
        if i == 0:
            x_train = tmp_x
            y_train = tmp_y
        else:
            x_train = np.vstack((x_train, tmp_x))
            y_train = np.hstack((y_train, tmp_y))

    return x_train, y_train


def extend_argmenter(x, mode='someof'):

    aug_x = []
    for i in range(x.shape[0]):
        each_x = x[i]
        if mode == 'someof':
            some_aug = iaa.SomeOf(2, [
                iaa.AdditiveGaussianNoise(scale=[0, 0.25*1]), # GaussianNoise
                iaa.AdditiveLaplaceNoise(scale=[0, 0.25*1]),  # LaplaceNoise
                #iaa.AdditivePoissonNoise(scale=[0, 0.25*1]),  # PoissonNoise
                #iaa.LogContrast(G, PCH)  # 明暗
                #iaa.LinearContrast(S, PCH) # 彩度変換 = 鮮鋭化?
                #iaa.AveragePooling(K, KS)  # 平滑化
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # Edge 強調?
                iaa.Invert(0.05, per_channel=True), # invert color channels
            ])
            aug_x.append(some_aug.augment_images(each_x))
        elif mode == 'one':
            aug = iaa.AdditiveGaussianNoise(scale=[0, 0.25*1])
            aug_x.append(aug.augment_image(each_x))
        elif mode == 'all':
            seq_aug = iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale=[0, 0.25*255]), # GaussianNoise
                iaa.AdditiveLaplaceNoise(scale=[0, 0.25*255]),  # LaplaceNoise
                iaa.AdditivePoissonNoise(scale=[0, 0.25*255]),  # PoissonNoise
                #iaa.LogContrast(G, PCH)  # 明暗
                #iaa.LinearContrast(S, PCH) # 彩度変換 = 鮮鋭化?
                #iaa.AveragePooling(K, KS)  # 平滑化
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # Edge 強調?
                iaa.Invert(0.05, per_channel=True), # invert color channels
            ])
            aug_x.append(seq_aug.augment_images(each_x))

    return np.array(aug_x)

def aug_img_plt(x):

    plt.figure(figsize=(12, 6))

    for i in range(10):
        plt.subplot(2, 5, i+1)
        
        plt.imshow(x[i])

        # 軸 非表示
        plt.axis(False)

    plt.show()
    
if __name__ == '__main__':

    current_location = os.path.abspath(__file__)  # このファイルの絶対パスを取得
    cwd, base_name = os.path.split(current_location)  # path と ファイル名に分割
    file_name, _ = os.path.splitext(base_name)  # ファイル名と拡張子を分離
    print("current location : ", cwd, ", this file : ", file_name)

    cnn_dir = os.path.dirname(cwd)
    
    # 少ないデータに対して水増しを行いたいので smaller を選択
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    train_dir = os.path.join(data_dir, "train")  # global
    validation_dir = os.path.join(data_dir, "validation")  # global
    print("train data is in ... ", train_dir)
    print("validation data is in ...", validation_dir)

    
    #validation_generator = validationDataGenerator(validation_dir)
    #train_generator = trainDataGenerator(train_dir)

    #data_checker, label_checker = next(train_generator)
    #print("data shape : ", data_checker.shape)
    #print("label shape : ", label_checker.shape)

    x_train, y_train = train_data_stocker(train_dir)
    x_validation, y_validation = validation_data_stocker(validation_dir)

    print("x_train.shape: ", x_train.shape)
    print("y_train.shape:" ,y_train.shape)
    print("x_validation.shape: ", x_validation.shape)
    print("y_validation.shape: ", y_validation.shape)

    x_train_aug = extend_argmenter(x_train, mode='one')
    print("x_train_aug.shape: ", x_train_aug.shape)

    aug_img_plt(x_train_aug)
