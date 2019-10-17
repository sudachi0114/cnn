
# train / test1 dataset から
#   300枚のデータセット作り

import os, shutil

# directory -----
cwd = os.getcwd()
print("cwd : ", cwd)

cnn_dir = os.path.dirname(cwd) # ../cnn
print("cnn_dir : ", cnn_dir) 

origin_data_dir = os.path.join(cnn_dir, "dogs_vs_cats_origin")
print("origin_data_dir : ", origin_data_dir)

# データセットを小さくしたディレクトリへの PATH
base_dir = os.path.join(cnn_dir, "dogs_vs_cats_mid300")
print("base_dir : ", base_dir)
os.makedirs(base_dir, exist_ok=True)

print('-*-'*10)

# train / valid / test data を配置する dir
data_split = ["train", "validation", "test"]

# dataset size (per one class) -----
train_amount = 150
validation_amount = 50
test_amount = 50

validation_begin = train_amount
validation_end = train_amount + validation_amount

test_begin = validation_end
test_end = validation_end + test_amount

for name in data_split:
    print("make directry : {}..".format(name))
    target_dir = os.path.join(base_dir, name)
    print("target_dir : ", target_dir)
    os.makedirs(target_dir, exist_ok=True)

    class_label = ["cat", "dog"]

    for class_name in class_label:
        # train/cats or dogs
        print("make directry : {}/{}..".format(name, class_name))
        target_class_dir = os.path.join(target_dir, class_name)
        print("target_class_dir :", target_class_dir)
        os.makedirs(target_class_dir, exist_ok=True)

        pic_name_list = []
        if name == "train":
            print("Amount of {}/{} pictures is : {}".format(name, class_name, train_amount) )
            print("train range is {} to {}".format(0, train_amount))
            for i in range(train_amount):
                pic_name_list.append("{}.{}.jpg".format(class_name, i))

        elif name == "validation":
            print("Amount of {}/{} pictures is : {}".format(name, class_name, validation_amount) )
            print("validation range is {} to {}".format(validation_begin, validation_end))
            for i in range(validation_begin, validation_end):
                pic_name_list.append("{}.{}.jpg".format(class_name, i))

        elif name == "test":
            print("Amount of {}/{} pictures is : {}".format(name, class_name, test_amount))
            print("test range is {} to {}".format(test_begin, test_end))
            for i in range(test_begin, test_end):  # (75 ~ 99) の 25枚分
                pic_name_list.append("{}.{}.jpg".format(class_name, i))

        print("Copy name : {}/{} | pic_name_list : {}".format(name, class_name, pic_name_list))

        for pic_name in pic_name_list:
            copy_src = os.path.join(origin_data_dir, pic_name)
            copy_dst = os.path.join(target_class_dir, pic_name)
            shutil.copy(copy_src, copy_dst)

        print("Done.")

    print('-*-'*10)
