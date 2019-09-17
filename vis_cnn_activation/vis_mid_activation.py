
# CNN の中間層の出力を可視化する。
#   model と weight が一緒の h5 file に保存されている必要がありそう..

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image

from keras.models import load_model

from keras import models

def load_img(img_path, input_size=150, show=False):

    # load image -----
    img = image.load_img(img_path, target_size=(input_size, input_size))
    img_tensor = image.img_to_array(img)
    img_tensor /= 255.0
    print("img_tensor.shape : ", img_tensor.shape)
    #print(img_tensor)

    if show:
        plt.imshow(img_tensor)
        plt.show()

    return img_tensor


def main(img_path, model_location, img_tensor):

    # confirm image path -----
    print("Datas is ... ", img_path)

    # confirm model location ----
    print("model location is ... ", model_location)
    
    # load model -----
    model = load_model(model_location)

    model.summary()

    # 出力の 8つの層から出力を抽出
    #layer_outputs = [layer.output for layer in model.layers[:8]]
    layer_outputs = []
    for layer in model.layers[:8]:
        layer_outputs.append(layer.output)


    # 特定の入力をもとに、その出力を返すモデルを作成
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # 特定の入力 = img_tensor
    #   img_tensor を読み込む前に このCNN に適する形に変換
    #       具体的には (150, 150, 3) => (1, 150, 150, 3) にする。

    img_tensor = np.expand_dims(img_tensor, axis=0)
    print("img_tensor shape (expanded) : ", img_tensor.shape)
    
    activations = activation_model.predict(img_tensor)

    # 例えば、最初の畳み込み層の活性化は次のようになる。
    first_layer_activation = activations[0]
    print("first_layer_actuvation's shape : ", first_layer_activation.shape)

    plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
    plt.show()


if __name__ == '__main__':

    # directory -----
    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")

    """
    img_dir = os.path.join(data_dir, "train")
    print("Datas are in ", img_dir)

    viable_img_list = os.listdir(img_dir)
    print("please choose one image below :\n", viable_img_list)

    choice = input(">>>")

    img_path = os.path.join(img_dir, choice)
    """
    
    img_dir = os.path.join(data_dir, "train" , "cat")  # 勝手にcat で固定したのであとで選択できるように直す。
    image_path = os.path.join(img_dir, "cat.3.jpg")

    img_tensor = load_img(image_path, show=True)

    # ------------------------------------------------


    # select model -----
    log_dir = os.path.join(cnn_dir, "log")

    """
    log_list = os.listdir(log_dir)
    print("\nplease chose log below : \n", log_list)
    choice = input(">>> ")
    
    child_log_dir = os.path.join(log_dir, choice)
    print("You choose", child_log_dir, "\n")

    child_log_list = os.listdir(child_log_dir)
    print("\nplease chose saved model below : \n", child_log_list)
    saved_model = input(">>> ")x

    model_location = os.path.join(child_log_dir, saved_model)
    print("\nYour model is ", model_location, "\n")
    """

    child_log_dir = os.path.join(log_dir, "binary_classifer_log")
    model_location = os.path.join(child_log_dir, "binary_dogs_vs_cats_model.h5")
    
    main(image_path, model_location, img_tensor)
