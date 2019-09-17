
# 各チャネルの活性をマッピングして描画するプログラム:

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model, Model

def load_img(img_location, input_size=150, show=True, expand=False):

    # load image -----
    img = image.load_img(img_location, target_size=(input_size, input_size))
    img_tensor = image.img_to_array(img)
    img_tensor /= 255.0
    print("img_tensor's shape : ", img_tensor.shape)

    if show:
        plt.imshow(img_tensor)
        plt.show()

    if expand:  # CNN に入力するために batch 次元分を拡張
        img_tensor = np.expand_dims(img_tensor, axis=0)
        print("-> expand img_tensor's shape : ", img_tensor.shape)

    return img_tensor

def generate_model(model_location):

    # load model -----
    print("\nload model...")
    loaded_model = load_model(model_location)
    print("Done.\n")

    loaded_model.summary()

    # get 8 layer's activation (output)
    layers_outputs = [layer.output for layer in loaded_model.layers[:8]]

    # generate "activation return" model (Model = keras.models.Model)
    print("\ngenerate keras.models.Model...")
    model = Model(inputs=loaded_model.input, outputs=layers_outputs)
    print("Done.")

    return model


def main(img_tensor, model):

    # confirm -----
    print("got img_tensor that shape is ", img_tensor.shape)
    print("got model object ...", model)

    # get each layer's name ---
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    print("layer_names : ", layer_names)

    imgs_per_row = 16

    # model に対する順伝播の出力を格納する変数
    activations = model.predict(img_tensor)

    # show feature maps -----
    for layer_name, layer_activation in zip(layer_names, activations):
        # 特徴マップ (出力) に含まれている特徴量 (フィルタ) の数 (?)
        feature_num = layer_activation.shape[-1]  # .shape で出てくる配列の最後尾を参照

        # 特徴マップの形状 (1, f_width, f_height, feature_num): いま width = height = size と仮定
        size = layer_activation.shape[1]

        # 出力をタイル状に表示するときの形状を計算する
        col_num = feature_num // imgs_per_row
        display_grid = np.zeros((size*col_num, imgs_per_row*size))

        # 各フィルタをタイル状に並べ、大きな1枚の絵として表示
        for col in range(col_num):
            for row in range(imgs_per_row):
                channel_img = layer_activation[0, :, :, col*imgs_per_row+row]

                # フィルタの見た目をよくするための後処理は (必要だと感じたら) あとで行う
                channel_img -= channel_img.mean()
                channel_img /= channel_img.std()
                channel_img *= 64
                channel_img += 128
                channel_img = np.clip(channel_img, 0, 255).astype('uint8')
                display_grid[col*size:(col+1)*size, row*size:(row+1)*size] = channel_img

        # grid を表示
        scale = 1. / size
        plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))

        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


if __name__ == '__main__':

    # dorectory -----
    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    print("cnn_dir : ", cnn_dir)

    # img_dir -----
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    img_dir = os.path.join(data_dir, "train", "cat")  # とりあえず train/ cat に固定

    img_location = os.path.join(img_dir, "cat.3.jpg")

    # model_dir -----
    log_dir = os.path.join(cnn_dir, "log")
    child_log_dir = os.path.join(log_dir, "binary_classifer_log")
    model_location = os.path.join(child_log_dir, "binary_dogs_vs_cats_model.h5")

    # get img_tensor -----
    img_tensor = load_img(img_location, show=False, expand=True)

    # load model -----
    model = generate_model(model_location)

    main(img_tensor, model)
