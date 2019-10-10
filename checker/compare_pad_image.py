
# データの水増しの為に幾何変形された画像を見てみる。
#   元データとの比較
#   3枚を 3回生成して比較

import os
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
cnn_dir = os.path.dirname(cwd)

data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
train_dir = os.path.join(data_dir, "train")

from keras.preprocessing.image import ImageDataGenerator

def main(input_size=150, batch_size=3, shuffle=False):

    # 画像の表示
    plt.figure()

    datagen = ImageDataGenerator(rescale=None,
                                 zoom_range=0.5  # 画像の内側をランダムにズーム
    )

    for i in range(3):  # 3回生成
        images = []
        labels = []

        print("gen: [", i, "]")
        image_generator = datagen.flow_from_directory(train_dir,
                                                  target_size=(input_size, input_size),
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  class_mode='binary')


        images, labels = next(image_generator)

        print("images.shape : ", images.shape)
        print("labels.shape : ", labels.shape)

        #plt.subplot(3, 1, i+1)

        for j in range(images.shape[0]):
            plt.subplot(1, 3, j+1)

            plt.imshow((images[j]).astype(np.uint8))

            # 軸 非表示
            plt.axis(False)

            # 正解ラベルを上部に表示
            if labels[i] == 0:
                name = "cat"
            else:
                name = "dog"

            plt.title(name)

        plt.show()
    

if __name__ == '__main__':
    main()
