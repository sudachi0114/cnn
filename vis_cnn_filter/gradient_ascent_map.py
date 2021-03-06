
# 多層に及ぶ各フィルタを可視化
#   各層から 最初の 64 個 のフィルタをタイル表示させてみる。
#   層は (block1_conv1, block2_conv1, block3_conv1, block4_conv1)
#   の 4つに関して調べる。

from keras.applications import VGG16
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

# 画像を表示するのに都合のいいように後処理する関数
def deprocess_img(x):
    
    # テンソルの正規化(平均0, 分散 0.1)
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # [0, 1] でクリッピング
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGB 配列に変換
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x
    

# 層とフィルタの指定を受けて、応答パターンを浸透させた画像を返す関数 (?)
def generate_img(layer_name, filter_idx, input_size=150, ch=3, iter_num=40):

    # VGG16 のモデルを定義
    model = VGG16(weights='imagenet',
                  include_top=False)

    # 特定の layer を名前で指定
    layer_output = model.get_layer(layer_name).output
    # 特定の filter を番目で指定
    loss = K.mean(layer_output[:, :, :, filter_idx])

    # 勾配計算
    grads = K.gradients(loss, model.input)[0]

    # 勾配の正規化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 入力画像に基づいて、損失値と勾配を計算する keras backend 関数
    loss_grads_gen = K.function([model.input], [loss, grads])

    # ランダムに初期化した画像を最初の入力として使用
    input_img = np.random.random((1, input_size, input_size, ch))*20 + 128.0

    # 勾配上昇法を 指定回数分実行
    print("processing {} | {} ...".format(layer_name, filter_idx), end="")
    lr = 1
    for i in range(iter_num):
        loss_val, grads_val = loss_grads_gen([input_img])
        input_img += grads_val*lr
    print("<= Done.")

    img = input_img[0]

    return deprocess_img(img)

# test
def test():
    
    print("\nstart testing program...")
    img = generate_img('block3_conv1', 0, input_size=64)
    plt.imshow(img)
    plt.show()

# 層とフィルタ名を繰り返し指定し、帰ってきた画像をタイル表示
def main(filter_num=64, margin=5):

    print("\nstart main program...")

    # 調べる layer たち
    #layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']  # これを全て見るには結構な計算量がある..
    layers = ['block3_conv1']

    for layer_name in layers:
        results = np.zeros((8*filter_num+7*margin, 8*filter_num+7*margin, 3))  # 出力タイルの初期化 (あとでタイルを埋める)

        for i in range(8):  # results grid の行を順番に処理
            for j in range(8):  # reults grid の列を順番に処理
                # layer_name の filter i+(j*8) のパターンを生成
                filter_img = generate_img(layer_name, i+(j*8), input_size=filter_num)  # 引数 : (layer_name, filter_idx, input_size)

                # results grid の矩形(タイル) に結果を配置
                horizontal_start = i*filter_num + i*margin
                horizontal_end = horizontal_start + filter_num
                vertical_start = j*filter_num + j*margin
                vertical_end = vertical_start + filter_num

                results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_img

        # results grid を表示 (layer 毎?)
        plt.figure(figsize=(20,20))
        plt.imshow(results.astype('uint8'))  # 参考書にはないが、ここで型変換してやらないと表示できない (画素がほとんど白)
        plt.show()

if __name__ == '__main__':

    #test()

    main()
