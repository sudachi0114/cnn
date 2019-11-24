
# 目標: layer は固定で、filter をたくさん出す

from keras.applications import VGG16
import keras.backend as K

import tensorflow as tf
print("GPU available: ", tf.test.is_gpu_available())
if tf.test.is_gpu_available():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

import os, platform
import numpy as np
import matplotlib.pyplot as plt


def model_hander(input_size=224, ch=3, iter_num=40):
    # [memo]
    #   描画された filter がノイズっぽい場合は
    #       iteration の回数が足りない可能性..

    model = VGG16(weights='imagenet',
                  include_top=False)

    model.summary()
    print("\n")

    # select layer & filter number ----------
    layer_name = 'block5_conv3'
    filter_index = 511
    print("\nget filter: {} in layer: {}".format(filter_index, layer_name))

    # target層の n番目のフィルタの活性化を最大化する損失関数を定義
    layer_output = model.get_layer(layer_name).output
    print("\nlayer output: ", layer_output)  # Tensor("block5_conv3/Relu:0", shape=(None, None, None, 512), dtype=float32)
    print("type(layer output): ", type(layer_output))  # <class 'tensorflow.python.framework.ops.Tensor'>

    loss = K.mean(layer_output[:, :, :, filter_index])
    print("\nloss: ", loss)  # loss:  Tensor("Mean:0", shape=(), dtype=float32)
    print("type(loss): ", type(loss))  # <class 'tensorflow.python.framework.ops.Tensor'>

    # 入力に関する損失関数の勾配を取得
    #   gradients の戻り値はテンソルのリスト (今回の場合は サイズ1 のリスト)
    #   このため、リストの 0番目の要素だけを持つようにしている。
    grads = K.gradients(loss, model.input)[0]
    print("\ngrads all: ", K.gradients(loss, model.input))  # grads all:  [<tf.Tensor 'gradients_1/block1_conv1/convolution_grad/Conv2DBackpropInput:0' shape=(None, None, None, 3) dtype=float32>]
    print("\ngrads: ", grads)  # Tensor("gradients/block1_conv1/convolution_grad/Conv2DBackpropInput:0", shape=(None, None, None, 3), dtype=float32)
    print("type(grads): ", type(grads))  # <class 'tensorflow.python.framework.ops.Tensor'>

    
    # 勾配の正規化
    #   1e-5 を足しているのは 0 除算回避のため
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 入力画像に基づいて、損失テンソルと勾配テンソルを計算する。
    #   iterate は引数に Numpy のテンソル (サイズ 1 のテンソルのリスト) をとり
    #   戻り値に 2つの Numpy テンソル (損失値と勾配値) のリストを返す関数として
    #   keras の backend 関数で定義。
    iterate = K.function([model.input], [loss, grads])
    print("\niter: ", iterate)  #  <tensorflow.python.keras.backend.EagerExecutionFunction object at 0x13d7f6f60>
    print("type(iter): ", type(iterate)) # <class 'tensorflow.python.keras.backend.EagerExecutionFunction'>
    

    # test
    #loss_value, grads_value = iterate([np.zeros((1, input_size, input_size, 3))])

    # input として ノイズが入ったグレースケール画像を使用
    input_img = np.random.random((1, input_size, input_size, ch))*20 + 128.

    print("\nshowing input image...")
    plt.imshow(input_img[0]/255)
    plt.show()

    # 勾配上昇法を 40 step 実行
    print("started gradient accendant...")
    lr = 1  # 各勾配の更新の大きさ
    for i in range(iter_num):
        # 損失関数と勾配を計算
        loss_val, grads_val = iterate([input_img])
        #print("  loss_val: ", loss_val)
        #print("  grads_val: ", grads_val)
        input_img += grads_val*lr
        print("  => Now step {} has done.".format(i+1))

    img = input_img[0]

    return img


# 結果として得られる画像テンソルは
#   形状が (1, 150, 150, 3) の float型のテンソル
#   このテンソルに含まれている値は [0, 255] の範囲の整数ではない可能性がある。
#       そのため、このテンソルの後処理を行なって表示可能な画像に変換する。
# そのための関数を以下で定義
def deprocess_img(x):

    # テンソルを正規化: 平均を 0, 標準偏差を 0.1 にする。
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # [0, 1] で cliping
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGB 配列に変換
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x


# 描画関数 ----------
def display(img):

    plt.imshow(img)
    if platform.system() == 'Linux':
        cwd = os.getcwd()
        pics_dir = os.path.join(cwd, "pictures")
        os.makedirs(pics_dir, exist_ok=True)
        plt.savefig(os.path.join(pics_dir, "gradient_ascent_pic.png"))
        print("Save figure in ", pics_dir)
    else:
        plt.show()

    
if __name__ == '__main__':
    img = model_hander()
    dep_img = deprocess_img(img)
    display(dep_img)
