
# Kaggle から download した train / test1 dataset の整理をするプログラム

import os, shutil

# cnn dir までの PATH
cwd = os.getcwd()
print("cwd : ", cwd)

cnn_dir = os.path.dirname(cwd) # .../cnn
print("cnn_dir : ", cnn_dir) 

origin_data_dir = os.path.join(cnn_dir, "train")
print("origin_data_dir : ", origin_data_dir)

# データセットを小さくしたディレクトリへの PATH
base_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
print("base_dir : ", base_dir)
os.makedirs(base_dir, exist_ok=True)

print('-*-'*10)

# train / valid / test data を配置する dir
data_split = ["train", "validation", "test"]

for name in data_split:
    print("make directry : {}..".format(name))
    target_dir = os.path.join(base_dir, name)
    print(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    class_label = ["cats", "dogs"]

    for class_name in class_label:
        # train/cats or dogs
        print("make directry : {}/{}..".format(name, class_name))
        target_class_dir = os.path.join(target_dir, class_name)
        print("target_class_dir :", target_class_dir)
        os.makedirs(target_class_dir, exist_ok=True)

    print('-*-'*10)
