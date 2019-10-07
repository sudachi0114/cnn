
# 学習用の画像のチェック
#   dogs_vs_cats_smaller/test 検視用
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator


def main():

    input_size = 150
    batch_size = 10

    # directory -----
    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    base_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")

    train_dir = os.path.join(base_dir, "test")

    train_datagen = ImageDataGenerator(rescale=None)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(input_size, input_size),
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        class_mode='binary')

    images = []
    labels = []

    
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

    #print("(images) : \n", images)
    #print("(labels) : \n", labels)

    # batch_size 分の画像を表示する。
    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(hspace=0.5)

    for i in range(images.shape[0]):
        plt.subplot(5, 10, i+1)
        plt.imshow((images[i]).astype(np.uint8))
        plt.axis(False)
        
        # x軸 非表示
        #plt.tick_params(labelbottom=False)

        # y軸 非表示
        #plt.tick_params(labelleft=False)

        # 正解ラベルを表示
        if labels[i] == 0:
            class_name = "cat"
        elif labels[i] == 1:
            class_name = "dog"
        plt.title("[{0}] {1}".format(i, class_name))

    plt.show()

if __name__ == '__main__':

    main()
