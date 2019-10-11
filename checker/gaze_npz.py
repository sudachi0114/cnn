
# npz 形式で保存したデータの検視

import os
import numpy as np
import matplotlib.pyplot as plt

def read_npz(file_path):
    
    npz = np.load(file_path)

    print(npz)  # <numpy.lib.npyio.NpzFile object at 0x106dccf28>
    print(npz.files)  # ['train_data', 'train_label']

    train_data, train_label = npz['train_data'], npz['train_label']

    return train_data, train_label

def display(image, label):

    image /= 255.0
    plt.figure(figsize=(12,6))
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.90)

    for i in range(len(label)):
        plt.subplot(10, 10, i+1)
        plt.imshow(image[i])
        plt.axis(False)
        plt.title("class: {}".format(label[i]))

    plt.show()

if __name__ == '__main__':

    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")

    file_path = os.path.join(data_dir, "train.npz")

    train_data, train_label = read_npz(file_path)

    print("train_data.shape: ", train_data.shape)  # (100, 224, 224, 3)
    print("train_label.shape: ", train_label.shape)  # (100,)

    display(train_data, train_label)
