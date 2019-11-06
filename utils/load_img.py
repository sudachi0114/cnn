
# Iterator で返されるのが嬉しくないので、自家製 load_img を作成
import os, sys
sys.path.append(os.pardir)

import numpy as np
from PIL import Image

def load_img():

    # パスの決定は外部ファイルに委託
    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)

    data_dir = os.path.join(prj_root, "dogs_vs_cats_smaller")
    train_data_dir = os.path.join(data_dir, "train")
    cat_train_data_dir = os.path.join(train_data_dir, "cat")

    # ここから load_img 処理 ----
    #   ディレクトリを受け取ってその中身をスタックするだけの関数に変更
    pic_list = os.listdir(cat_train_data_dir)

    train_data = []
    for picture in pic_list:
        target = os.path.join(cat_train_data_dir, picture)
        img_obj = Image.open(target)
        img_resize = img_obj.resize((224, 224))
        img_arr = np.asarray(img_resize)
        train_data.append(img_arr)

    train_data = np.array(train_data)

    print(train_data.shape)
    print(train_data[1].shape)
    # return (train_) data する。
    #   label はここで返すのがいいのか
    #   外部ファイルがいいのか..??






if __name__ == '__main__':
    load_img()