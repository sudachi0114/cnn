
# Iterator で返されるのが嬉しくないので、自家製 load_img を作成
import os, sys
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ignore_list = [".DS_Store"]

def load_img(fpath, array_size):

    img_obj = Image.open(fpath)

    resized_img = img_obj.resize((array_size, array_size))
    img_array = np.asarray(resized_img)

    return img_array



def loadImageFromDir(target_dir, input_size):

    pic_list = os.listdir(target_dir)

    for canditate in ignore_list:
        if canditate in pic_list:
            pic_list.remove(canditate)

    sorted_pic_list = sorted(pic_list)
    print("found {} images ...".format(len(pic_list)))

    img_arrays = []
    for picture in sorted_pic_list:
        target = os.path.join(target_dir, picture)
        img_arr = load_img(target, input_size)
        img_arrays.append(img_arr)

    img_arrays = np.array(img_arrays)

    assert img_arrays.shape[0] == len(pic_list)

    return img_arrays


def inputDataCreator(target_dir, input_size, normalize=False):

    class_list = os.listdir(target_dir)

    for canditate in ignore_list:
        if canditate in class_list:
            class_list.remove(canditate)

    print("found {} classes ...".format(len(class_list)))

    img_arrays = []
    labels = []

    each_class_img_arrays = []
    label = []

    sorted_class_list = sorted(class_list)
    for class_num, class_name in enumerate(sorted_class_list):
        each_class_data_dir = os.path.join(target_dir, class_name)
        print("processing class {} as {} ".format(class_num, class_name), end="")

        each_class_img_arrays = loadImageFromDir(each_class_data_dir, input_size)
        label = np.full(each_class_img_arrays.shape[0], class_num)

        if img_arrays == []:
            img_arrays = each_class_img_arrays
        else:
            img_arrays = np.vstack((img_arrays, each_class_img_arrays))

        if label == []:
            labels = label
        else:
            labels = np.hstack((labels, label))

    if normalize:
        img_arrays = img_arrays / 255

    img_arrays = np.array(img_arrays)
    labels = np.array(labels)

    assert img_arrays.shape[0] == labels.shape[0]

    return img_arrays, labels


def display(img_array, label):

    plt.imshow(img_array)
    plt.title("label: {}".format(label))
    plt.show()





if __name__ == '__main__':

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)

    data_dir = os.path.join(prj_root, "dogs_vs_cats_smaller")
    train_data_dir = os.path.join(data_dir, "train")
    cat_train_data_dir = os.path.join(train_data_dir, "cat")
    sample_cat = os.path.join(cat_train_data_dir, "cat.0.jpg")

    import argparse

    parser = argparse.ArgumentParser(description="画像読み込みに関する自家製ミニマルライブラリ (速度はあまりコミットしてないです..)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--display", type=int, default=99999)

    args = parser.parse_args()

    if args.test:
        print("\ntesting load_img():")
        single_img_array = load_img(sample_cat, 224)
        print("  result: ", single_img_array.shape)

        print("\ntesting loadImageFromDir():")
        train_cat_datas = loadImageFromDir(cat_train_data_dir, 224)
        print("  result: ", train_cat_datas.shape)

        print("\ntesting inputDataCreator(train_data_dir, 224, normalize=True):")
        data, label = inputDataCreator(train_data_dir, 224, normalize=True)
        print("  result (data shape) : ", data.shape)
        print("    data: \n", data[0])
        print("  result (label shape):", label.shape)
        print("    label: \n", label)

        print("\ntesting inputDataCreator(train_data_dir, 224, normalize=False:")
        data, label = inputDataCreator(train_data_dir, 224, normalize=False)
        print("  result (data shape) : ", data.shape)
        print("    data: \n", data[0])
        print("  result (label shape): ", label.shape)
        print("    label: \n", label)


    if args.time:
        print("\ntesting loadImageFromDir() in large data:")
        origin_dir = os.path.join(prj_root, "dogs_vs_cats_origin")
        import time
        start = time.time()
        large_test_datas = loadImageFromDir(origin_dir, 224)
        print("  result: ", large_test_datas.shape)
        print("elapsed time: ", time.time() - start, " [sec]")


    if args.display != 99999:
        print("read & display...")
        data, label = inputDataCreator(train_data_dir, 224)

        display(data[args.display], label[args.display])

    print("Done.")

