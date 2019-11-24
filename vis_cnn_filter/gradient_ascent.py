
# 各フィルタが応答する資格パターンを表示
#   入力空間で勾配上昇法 (gradient ascent) を用いる。
#       空の入力空間から始めて、 CNN の入力画像の値に
#       勾配降下法を適用することで、特定のフィルタの応答を最大化する (?)
#
#       この結果として得られる入力画像が
#       選択されたフィルタの応答性が最も高いものになる。

from keras.applications import VGG16
import keras.backend as K

import tensorflow as tf
print("GPU available: ", tf.test.is_gpu_available())
if tf.test.is_gpu_available():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    #K.set_session(sess)

import os, platform
import numpy as np
import matplotlib.pyplot as plt

def model_hander(input_size=150, ch=3, iter_num=40):

    model = VGG16(weights='imagenet',
                  include_top=False)

    model.summary()

    # ここで "特定のレイヤ" を選択 (今回は block3_conv1)
    #layer_name = 'block3_conv1'
    layer_name = 'block5_conv3'
    # "特定のフィルタ" を選択 (今回は 0番目)
    filter_index = 12

    # target層の n番目のフィルタの活性化を最大化する損失関数を定義
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # 入力に関する損失関数の勾配を取得
    #   gradients の戻り値はテンソルのリスト (今回の場合は サイズ1 のリスト)
    #   このため、リストの 0番目の要素だけを持つようにしている。
    grads = K.gradients(loss, model.input)[0]

    # 勾配の正規化
    #   1e-5 を足しているのは 0 除算回避のため
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 入力画像に基づいて、損失テンソルと勾配テンソルを計算する。
    #   iterate は引数に Numpy のテンソル (サイズ 1 のテンソルのリスト) をとり
    #   戻り値に 2つの Numpy テンソル (損失値と勾配値) のリストを返す関数として
    #   keras の backend 関数で定義。
    iterate = K.function([model.input], [loss, grads])

    # test
    #loss_value, grads_value = iterate([np.zeros((1, input_size, input_size, 3))])

    # input として ノイズが入ったグレースケール画像を使用
    input_img = np.random.random ((1, input_size, input_size, ch))*20 + 128.

    # 勾配上昇法を 40 step 実行
    lr = 1  # 各勾配の更新の大きさ
    for i in range(iter_num):
        # 損失関数と勾配を計算
        loss_val, grads_val = iterate([input_img])
        # 損失値が大きくなる方向に(?) "入力画像を" 調整
        input_img += grads_val*lr
        print("Now step {} has done.".format(i+1))

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

# filter を可視化するための関数
#   正直これだけ書いてもよかったのかもしれない
#    あとで、reload_model の filter もこれで見てみる (TODO:)
def plt_filter(img):

    plt.imshow(img)
    if platform.system() == 'Linux':
        cwd = os.getcwd()
        pics_dir = os.path.join(cwd, "pictures")
        os.makedirs(pics_dir, exist_ok=True)
        plt.savefig(os.path.join(pics_dir, "gradient_ascent_pic.png"))
        print("Save figure in ", pics_dir)
    else:
        plt.show()
    # まじでか。

    
if __name__ == '__main__':
    img = model_hander()
    dep_img = deprocess_img(img)
    plt_filter(dep_img)
