
# 学習用の画像のチェック
import os

import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

def main(show_all=False, shuffle=True):

    input_size = 150
    batch_size = 10

    # directory -----
    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    base_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")

    train_dir = os.path.join(base_dir, "train")

    train_datagen = ImageDataGenerator(rescale=None)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(input_size, input_size),
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        class_mode='binary')

    images = []
    labels = []

    
    if not show_all:
        images, labels = next(train_generator)
    else:
        iter_num = train_generator.n // batch_size

        for i in range(iter_num):
            x, y = next(train_generator)
            if i == 0:
                images = x
                labels = y
            else:
                images = np.vstack((images, x))
                labels = np.hstack((labels, y))


    print("images.shape : ", images.shape)
    print("lebels.shape : ", labels.shape)

    print("(images) : \n", images)
    print("(labels) : \n", labels)

    # batch_size 分の画像を表示する。
    plt.figure(figsize=(15, 15))

    for i in range(images.shape[0]):
        if not show_all:
            img = plt.subplot(3, 4, i+1)
        else:
            img = plt.subplot(5, 20, i+1)
        #plt.imshow((images[i]*255).astype(np.uint8))
        plt.imshow((images[i]).astype(np.uint8))
        

        # x軸 非表示
        plt.tick_params(labelbottom=False)

        # y軸 非表示
        plt.tick_params(labelleft=False)

        # 正解ラベルを表示
        plt.title(labels[i])

    plt.show()

if __name__ == '__main__':
    #main(show_all=False)
    main(show_all=True, shuffle=False)
