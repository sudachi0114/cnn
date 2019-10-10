
# データの水増しの為に幾何変形された画像を見てみる。

import os
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
cnn_dir = os.path.dirname(cwd)

data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
train_dir = os.path.join(data_dir, "train")

from keras.preprocessing.image import ImageDataGenerator

def main(input_size=150, batch_size = 10, show_all=False, shuffle=True):

    datagen = ImageDataGenerator(rescale=None,
                                 rotation_range=40,  # 画像をランダムに回転させる(0 ~ 180)
                                 #width_shift_range=0.2,  # 画像をランダムに水平に平行移動させる
                                 #height_shift_range=0.2,  # 画像を垂直に平行移動させる
                                 #shear_range=0.2,  # 等積変形(shear変形) をランダムに適用
                                 #zoom_range=0.2,  # 画像の内側をランダムにズーム
                                 #horizontal_flip=True,  # 画像の半分を水平方向にランダムに反転
                                 #fill_mode='nearest'  # 新たに作成されたピクセルを近くの画素の情報を元に埋める
                                 channel_shift_range=5.0 # 色調をランダム変更
                                 )

    image_generator = datagen.flow_from_directory(train_dir,
                                                  target_size=(input_size, input_size),
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  class_mode='binary')

    images = []
    labels = []

    if not show_all:
        images, labels = next(image_generator)
    else:
        iter_num = image_generator.n // batch_size

        for i in range(iter_num):
            x, y = next(image_generator)
            if i == 0:
                images = x
                labels = y
            else:
                images = np.vstack((images, x))
                labels = np.hstack((labels, y))

    print("images.shape : ", images.shape)
    print("labels.shape : ", labels.shape)

    # 画像の表示
    plt.figure(figsize=(12, 6))

    for i in range(images.shape[0]):
        if not show_all:
            img = plt.subplot(3, 4, i+1)
        else:
            img = plt.subplot(5, 20, i+1)

        plt.imshow((images[i]).astype(np.uint8))

        # x軸 非表示
        plt.tick_params(labelbottom=False)
        
        # y軸 非表示
        plt.tick_params(labelleft=False)

        # 正解ラベルを上部に表示
        if labels[i] == 0:
            name = "cat"
        else:
            name = "dog"

        plt.title(name)

    plt.show()
    

if __name__ == '__main__':
    main(shuffle=False)
