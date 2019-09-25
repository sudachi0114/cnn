
# Kaggle から download した train dataset の整理をするプログラム

import os, shutil

# cnn dir までの PATH
cwd = os.getcwd()
print("cwd : ", cwd)

cnn_dir = os.path.dirname(cwd) # .../cnn
print("cnn_dir : ", cnn_dir) 

origin_data_dir = os.path.join(cnn_dir, "train")
print("origin_train_data_dir : ", origin_data_dir)

# 分類先のディレクトリへの PATH
base_dir = os.path.join(cnn_dir, "dogs_vs_cats_full")
print("base_dir : ", base_dir)
os.makedirs(base_dir, exist_ok=True)

print('-*-'*10)

# train / valid / test data を配置する dir
#   validation は validation sprit で指定しようかな
data_split = ["train", "validation"]

# meta 
data_amount = 12500
validation_split = 0.1

validation_num = int(data_amount*validation_split)
train_num = 12500 - validation_num

for name in data_split:

    # train / test 共通
    print("make directry : {}..".format(name))
    target_dir = os.path.join(base_dir, name)
    print("target_dir : ", target_dir)
    os.makedirs(target_dir, exist_ok=True)

    class_label = ["cat", "dog"]

    for class_name in class_label:
        # train or validation /cats or dogs
        print("make directry : {}/{}..".format(name, class_name))
        target_class_dir = os.path.join(target_dir, class_name)
        print("target_class_dir :", target_class_dir)
        os.makedirs(target_class_dir, exist_ok=True)

        pic_name_list = []
        if name == "train":
            print("Amount of {}/{} pictures is : {}".format(name, class_name, train_num))
            for i in range(train_num):
                pic_name_list.append("{}.{}.jpg".format(class_name, i))
            print("-> confirm : len(train_pic_name_list) = ", len(pic_name_list))

        elif name == "validation":
            print("Amount of {}/{} pictures is : {}".format(name, class_name, validation_num))
            for i in range(train_num, data_amount):
                pic_name_list.append("{}.{}.jpg".format(class_name, i))
            print("-> confirm : len(test_pic_name_list) = ", len(pic_name_list))
        
        #print("Copy name : {}/{} | pic_name_list : {}".format(name, class_name, pic_name_list))

        for pic_name in pic_name_list:
            copy_src = os.path.join(origin_data_dir, pic_name)
            copy_dst = os.path.join(target_class_dir, pic_name)
            shutil.copy(copy_src, copy_dst)

        print("Done.")

    print('-*-'*10)
