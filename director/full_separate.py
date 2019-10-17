
# Kaggle から download した train dataset の整理をするプログラム

import os, shutil

# cnn dir までの PATH
cwd = os.getcwd()
print("cwd : ", cwd)

cnn_dir = os.path.dirname(cwd) # .../cnn
print("cnn_dir : ", cnn_dir) 

origin_data_dir = os.path.join(cnn_dir, "train")  # 今 test1 dir にある画像使ってない..
print("origin_train_data_dir : ", origin_data_dir)

# 分類先のディレクトリへの PATH
base_dir = os.path.join(cnn_dir, "dogs_vs_cats_full")
print("base_dir : ", base_dir)
os.makedirs(base_dir, exist_ok=True)

print('-*-'*10)

# train / valid / test data を配置する dir
data_split = ["train", "validation", "test"]

# meta -----
# amount = 25000
#   train = 20000
#   val = 2500
#   test = 2500

class_label = ["cat", "dog"]

amount_per_class = len(os.listdir(origin_data_dir))//len(class_label)
train_num_per_class = 20000//len(class_label)
validation_num_per_class = 2500//len(class_label)
test_num_per_class = 2500//len(class_label)

#validation_begin = train_num
validation_end = train_num_per_class + validation_num_per_class

for name in data_split:

    # train / test 共通
    print("make directry : {}..".format(name))
    target_dir = os.path.join(base_dir, name)
    print("target_dir : ", target_dir)
    os.makedirs(target_dir, exist_ok=True)

    for class_name in class_label:
        # train or validation /cats or dogs
        print("make directry : {}/{}..".format(name, class_name))
        target_class_dir = os.path.join(target_dir, class_name)
        print("target_class_dir :", target_class_dir)
        os.makedirs(target_class_dir, exist_ok=True)

        pic_name_list = []
        if name == "train":
            print("Amount of {}/{} pictures is : {}".format(name, class_name, train_num_per_class))
            print("train range is {} to {}".format(0, train_num_per_class))
            for i in range(train_num_per_class):
                pic_name_list.append("{}.{}.jpg".format(class_name, i))
            print("-> confirm : len(train_pic_name_list) = ", len(pic_name_list))

        elif name == "validation":
            print("Amount of {}/{} pictures is : {}".format(name, class_name, validation_num_per_class))
            print("validation range is {} to {}".format(train_num_per_class, validation_end))
            for i in range(train_num_per_class, validation_end):
                pic_name_list.append("{}.{}.jpg".format(class_name, i))
            print("-> confirm : len(test_pic_name_list) = ", len(pic_name_list))

        elif name == "test":
            print("Amount of {}/{} pictures is : {}".format(name, class_name, amount_per_class-(train_num_per_class+validation_num_per_class)))
            print("test range is {} to {}".format(validation_end, amount_per_class))
            for i in range(validation_end, amount_per_class):
                pic_name_list.append("{}.{}.jpg".format(class_name, i))
            print("-> confirm : len(test_pic_name_list) = ", len(pic_name_list))

        #print("Copy name : {}/{} | pic_name_list : {}".format(name, class_name, pic_name_list))

        for pic_name in pic_name_list:
            copy_src = os.path.join(origin_data_dir, pic_name)
            copy_dst = os.path.join(target_class_dir, pic_name)
            shutil.copy(copy_src, copy_dst)

        print("Done.")

    print('-*-'*10)
