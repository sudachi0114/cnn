
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

# define
img_size = 224
channel = 3
input_shape = (img_size, img_size, channel)

# lim = 4

def dataLoader(select):
    
    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)

    data_dir = os.path.join(prj_root, "dogs_vs_cats_smaller", "train")
    if select == 'cat':
        child_dir = os.path.join(data_dir, "cat")
    elif select == 'dog':
        child_dir = os.path.join(data_dir, "dog")
    print(child_dir)

    img_list = os.listdir(child_dir)
    img_list = sorted(img_list)

    sample_location = os.path.join(child_dir, img_list[1])
    print(sample_location)

    pil_obj = Image.open(sample_location)
    pil_obj = pil_obj.resize((img_size, img_size))
    sample = np.array(pil_obj)
    print(sample.shape)

    singleDisplay(sample)

    sample = np.expand_dims(sample, axis=0)
    print(sample.shape)

    return sample



def singleDisplay(x):
    plt.imshow(x)
    plt.show()


def reloader():

    model = VGG16(input_shape=input_shape,
                weights='imagenet',
                include_top=False)

    model.summary()

    print(model.layers)
    print(len(model.layers))

    return model



def modeler(model, sample, lim):

    # 途中 (lim) までの層 (とその重み) を持つモデルを再定義
    layers = model.layers[:lim]
    print(len(layers))

    # 層を減らしたので red(uced)_model と呼ぶことにする。
    red_model = Sequential(layers)
    # いちいち lim したモデルの summary を表示したい場合はここのコメントを戻す。
    #red_model.summary()

    y_hw = red_model.predict(sample)
    #print(y_hw)
    print(y_hw.shape)

    # (1, 7, 7, 512) => (7, 7, 512)
    y_hw = np.squeeze(y_hw, axis=0)
    # (7, 7, 512) => (512, 7, 7)
    y_hw = y_hw.transpose(2, 0, 1)
    print(y_hw.shape)

    plt.figure(figsize=(10, 6))
    for i in range(30):
        plt.subplot(6, 5, i+1)
        #plt.imshow(y_hw[i])
        plt.imshow(y_hw[i], cmap='gray')
        plt.title(i)
        plt.axis(False)

    plt.show()

if __name__ == '__main__':

    sample = dataLoader('dog')
    model = reloader()
    
    for i in range(2, 19):
        print("\nnow is limit: {} ...".format(i))
        modeler(model, sample, lim=i)
