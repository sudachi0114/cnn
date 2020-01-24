
import os, sys
sys.path.append(os.pardir)
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



class MyImageDataHandler:

    def __init__(self, target_dir=None, batch_size=None):

        self.target_dir = target_dir
        self.batch_size = batch_size
        self.ignore_files = ['.DS_Store', '__pycache__']

        if target_dir is not None:
            clist = os.listdir(target_dir)
            self.class_list = self.dirsieve(clist)
        else:
            self.class_list = []


    def display(self, img_array, label):

        plt.imshow(img_array)
        plt.title("label: {}".format(label))
        plt.show()


    def load_img(self, fpath, array_size):
        """convert image file to numpy array by PIL
        
        # Args:
            fpath (str): 読み込みたいファイルのパス
            array_size (int): 画像読み込みの配列のサイズ (正方形を想定)

        # Returns:
            img_array (np.ndarray): 画像を np.ndarray で読み込んだ配列
        """

        img_obj = Image.open(fpath)
        resized_img = img_obj.resize((array_size, array_size))
        img_array = np.asarray(resized_img)

        return img_array


    def loadImageFromList(self, target_list, input_size):
        """list で指定した画像たちを読み込む

        # Args:
            target_list (list): 読み込みたい画像の list
                画像が格納されている location までの絶対 path
            input_size (int): 各画像を読み込みたい配列のサイズ (正方形を想定)
                => これを load_img() の array_size に渡す

        # Returns: img_arrays (np.ndarray): ディレクトリの中にあった画像をそれぞれ配列に変換して積み上げたもの
        """

        img_arrays = []
        for picture in target_list:
            img_arr = self.load_img(picture, input_size)
            img_arrays.append(img_arr)

        img_arrays = np.array(img_arrays)

        assert img_arrays.shape[0] == len(target_list)

        return img_arrays


    def loadImageFromDir(self, target_dir, input_size):
        """ディレクトリを指定して、その中にある画像を再帰的に読み込む

        # Args:
            target_dir (str): 読み込みたい画像が格納されているディレクトリ
            input_size (int): 各画像を読み込みたい配列のサイズ (正方形を想定)
                => これを load_img() の array_size に渡す

        # Returns: img_arrays (np.ndarray): ディレクトリの中にあった画像をそれぞれ配列に変換して積み上げたもの
        """

        pic_list = os.listdir(target_dir)
        pic_list = self.dirsieve(pic_list)
        print("found {} images ...".format(len(pic_list)))

        target_list = []
        for picture in pic_list:
            target_list.append( os.path.join(target_dir, picture) )

        img_arrays = self.loadImageFromList(target_list, input_size)

        return img_arrays




    def inputDataCreator(self, target_dir, input_size, normalize=False, one_hot=False):
        """CNN などに入力する配列を作成する
            keras ImageDataGenerator の Iterator じゃない版
            読み込んだ画像データが一気に memory 上に乗るので
            Out Of Memory Error に注意

        # Args:
            target_dir (str): 画像データのディレクトリ
            input_size (int): 各画像を読み込みたい配列のサイズ (正方形を想定)
                => これを load_img() の array_size に渡す
            normalize (bool): 画像を読み込む際に [0, 1] に変換するか
                False => [0, 255] (default)
                True => [0, 1]
            one_hot (bool): label を one-hot 表現に変換する
                False => 0 or 1 (default)
                True => [1, 0] or [0, 1]

        # Returns
            img_arrays (np.ndarray): 読み込んだ画像データの配列
            labels (np.ndarray): 読み込んだ画像に対する正解ラベル
        """

        if len(self.class_list) == 0:
            class_list = os.listdir(target_dir)
            class_list = self.dirsieve(class_list)
        print("found {} classes ...".format(len(self.class_list)))

        img_arrays = []
        labels = []

        each_class_img_arrays = []
        label = []

        for class_num, class_name in enumerate(class_list):
            each_class_data_dir = os.path.join(target_dir, class_name)
            print("processing class {} as {} ".format(class_name, class_num), end="")

            each_class_img_arrays = self.loadImageFromDir(each_class_data_dir, input_size)
            label = np.full(each_class_img_arrays.shape[0], class_num)

            if len(img_arrays) == 0:
                img_arrays = each_class_img_arrays
            else:
                img_arrays = np.vstack((img_arrays, each_class_img_arrays))

            if len(label) == 0:
                labels = label
            else:
                labels = np.hstack((labels, label))

        if normalize:
            img_arrays = img_arrays / 255

        img_arrays = np.array(img_arrays)
        labels = np.array(labels)

        # print("debug: ", labels[1])

        if one_hot:
            labels = np.identity(len(self.class_list))[labels.astype(np.int8)]

        assert img_arrays.shape[0] == labels.shape[0]

        return img_arrays, labels


    def dataSplit(self, data, label,
                  train_rate=0.6,
                  validation_rate=0.3,
                  test_rate=0.1):
        """順番に積み重なっているデータに対して 2class の場合に等分割する関数    
         # Args:
            data (np.ndarray): 画像データの配列
            label (np.ndarray): 画像データの正解ラベルの配列
            train_rate (float): 全データにおける train data の割合 (0 ~ 1) で選択
                                (default: 0.6 == 60%)
            valiation_rate (float): 全データにおける validation data の割合 (0 ~ 1) で選択
                                (default: 0.2 == 20%)
            test_rate (float): 全データにおける test data の割合 (0 ~ 1) で選択
                                (default: 0.1 == 10%)
            one_hot (bool): 引数の label が one_hot 表現か否か
                                (default: True)
        # Return:
            train_data, train_label
            validation_data, validation_label
            test data, test_label
        """

        if len(label.shape) == 1:  # when label is not one_hot
            class_num = len(set(label))
            one_hot = False
        else:
            class_num = len(label[0])
            one_hot = True
        print("\nData set contain {} class data.".format(class_num))

        amount = data.shape[0]
        print("Data amount: ", amount)
        each_class_amount = int(amount / class_num)
        print("Data each class data amount: ", each_class_amount)

        train_data, train_label = [], []
        validation_data, validation_label = [], []
        test_data, test_label = [], []


        # calucurate each data size
        train_size = int( each_class_amount*train_rate )  # 7
        validation_size = int( each_class_amount*validation_rate )  # 2
        test_size = int( each_class_amount*test_rate )  # 1

        print("train_size     : ", train_size)
        print("validation_size: ", validation_size)
        print("test_size      : ", test_size)


        # devide data -----
        for i in range(class_num):
            each_class_data = []
            each_class_label = []

            if one_hot:
                # label の i番目が i である index を取得
                idx = np.where(label[:, i] == 1)
                # print("condition: ", condition)
            else:
                # i である label の index を取得
                idx = np.where(label == i)
                # print("idx: ", idx)
            each_class_label = label[idx]
            each_class_data = data[idx]
            print("\nfound class {} data as shape: {}".format(i, each_class_data.shape))
            print("found class {} label as shape: {}".format(i, each_class_label.shape))


            # split data ----------
            each_train_data, each_validation_data, each_test_data = np.split(each_class_data,
                                                                             [train_size, train_size+validation_size])
            print("\ntrain_data at class{}: {}".format(i, each_train_data.shape))
            print("validation_data at class{}: {}".format(i, each_validation_data.shape))
            print("test_data at class{}: {}".format(i, each_test_data.shape))

            # 初回は代入, 2回目以降は (v)stack
            if len(train_data) == 0:
                train_data = each_train_data
            else:
                train_data = np.vstack((train_data, each_train_data))

            if len(validation_data) == 0:
                validation_data = each_validation_data
            else:
                validation_data = np.vstack((validation_data, each_validation_data))

            if len(test_data) == 0:
                test_data = each_test_data
            else:
                test_data = np.vstack((test_data, each_test_data))


            # split label ----------
            each_train_label, each_validation_label, each_test_label = np.split(each_class_label,
                                                                             [train_size, train_size+validation_size])
            print("\ntrain_label at class{}: {}".format(i, each_train_label.shape))
            print("validation_label at class{}: {}".format(i, each_validation_label.shape))
            print("test_label at class{}: {}".format(i, each_test_label.shape))

            # 初回は代入, 2回目以降は (v)stack
            if len(train_label) == 0:
                train_label = each_train_label
            else:
                if one_hot:
                    train_label = np.vstack((train_label, each_train_label))
                else:
                    train_label = np.hstack((train_label, each_train_label))

            if len(validation_label) == 0:
                validation_label = each_validation_label
            else:
                if one_hot:
                    validation_label = np.vstack((validation_label, each_validation_label))
                else:
                    validation_label = np.hstack((validation_label, each_validation_label))

            if len(test_label) == 0:
                test_label = each_test_label
            else:
                if one_hot:
                    test_label = np.vstack((test_label, each_test_label))
                else:
                    test_label = np.hstack((test_label, each_test_label))

            print("୨୧┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈୨୧")

        print("\n    ... end.\n")

        print("train_data.shape     : ", train_data.shape)
        print("validation_data.shape: ", validation_data.shape)
        print("test_data.shape      : ", test_data.shape)
        # print(test_label)

        # program test -----
        print("\ntest sequence... ")

        # train -----
        cls0_cnt = 0
        cls1_cnt = 0
        if one_hot:
            for i in range(len(train_label)):
                if train_label[i][0] == 1:
                    cls0_cnt += 1
                elif train_label[i][1] == 1:
                    cls1_cnt += 1
        else:
            cls0_cnt = len(train_label[train_label==0])
            cls1_cnt = len(train_label[train_label==1])
        assert cls0_cnt == cls1_cnt
        print("  -> train cleared.")

        # validation -----
        cls0_cnt = 0
        cls1_cnt = 0
        if one_hot:
            for i in range(len(validation_label)):
                if validation_label[i][0] == 1:
                    cls0_cnt += 1
                elif validation_label[i][1] == 1:
                    cls1_cnt += 1
        else:
            cls0_cnt = len(validation_label[validation_label==0])
            cls1_cnt = len(validation_label[validation_label==1])
        assert cls0_cnt == cls1_cnt
        print("  -> validation cleared.")

        # test -----
        cls0_cnt = 0
        cls1_cnt = 0
        if one_hot:
            for i in range(len(test_label)):
                if test_label[i][0] == 1:
                    cls0_cnt += 1
                elif test_label[i][1] == 1:
                    cls1_cnt += 1
        else:
            cls0_cnt = len(test_label[test_label==0])
            cls1_cnt = len(test_label[test_label==1])
        assert cls0_cnt == cls1_cnt
        print("  -> test cleared.\n")

        # returns
        splited_data = (train_data, train_label,
                        validation_data, validation_label,
                        test_data, test_label)
        return splited_data


    # directory iterator utilities below ----------
    def dirsieve(self, dir_list):

        for fname in self.ignore_files:
            if fname in dir_list:
                dir_list.remove(fname)
            dir_list = sorted(dir_list)

        return dir_list


    def glob_dir(self, target_dir, return_label=False):

        # print("target: ", target_dir)
        if len(self.class_list) == 0:
            raw_class_list = os.listdir(target_dir)
            self.class_list = self.dirsieve(raw_class_list)
            # print("class : ", self.class_list)

        abs_path_list = []
        labels = []
        self.amount = 0
        for i, cname in enumerate(self.class_list):
            sub_target_dir = os.path.join(target_dir, cname)
            # print(i, "|", sub_target_dir)

            sub_target_list = os.listdir(sub_target_dir)
            sub_target_list = self.dirsieve(sub_target_list)
            self.amount += len(sub_target_list)

            # create labels
            label = np.full(len(sub_target_list), i)
            labels = np.hstack((labels, label))

            # trainsform to abs_path list
            for i in range(len(sub_target_list)):
                abs_path = os.path.join(sub_target_dir, sub_target_list[i])
                abs_path_list.append(abs_path)

        if return_label:
            return sorted(abs_path_list), labels
        else:
            ## sub_target_list = dirsieve(sub_target_list)
            ## return sub_target_list
            return sorted(abs_path_list)



    def diriterator(self, target_dir, batch_size, return_label=False):

        if return_label:
            target_list, labels = self.glob_dir(target_dir, return_label=True)
            self.steps = self.amount // batch_size
            for i in range(self.steps):
                begin = i * batch_size  # 0, 10, 20, 30 ...
                end = begin + batch_size
                # print(target_list[begin:end])

                batch_flist = target_list[begin:end]
                batch_label = labels[begin:end]

                yield batch_flist, batch_label
        else:
            target_list = self.glob_dir(target_dir)
            self.steps = self.amount // batch_size
            # ここに shuffle 処理を条件付きで入れる

            for i in range(self.steps):
                begin = i * batch_size  # 0, 10, 20, 30 ...
                end = begin + batch_size
                # print(target_list[begin:end])

                batch_flist = target_list[begin:end]

                yield batch_flist


    # wrap diriterator due to convert recallable
    def recalliterator(self, target_dir, batch_size, epochs,
                       return_label=False):

        target_list = self.glob_dir(target_dir)  # amount 計算用
        steps = self.amount // batch_size

        self.total_call = steps*epochs

        if return_label:
            for i in range(self.total_call):
                # if iteration counts over iterable num
                #   re-instance iterator and generator start again
                if (i % steps == 0):
                    dit = self.diriterator(target_dir, batch_size, return_label=True)

                batch_flist, batch_label = next(dit)
                # print(batch)
                # print(i, "(from recalliterator)")

                yield batch_flist, batch_label

        else:
            for i in range(self.total_call):
                # if iteration counts over iterable num
                #   re-instance iterator and generator start again
                if (i % steps == 0):
                    dit = self.diriterator(target_dir, batch_size)

                batch_flist = next(dit)
                # print(batch)
                # print(i, "(from recalliterator)")

                yield batch_flist


    def flowfromdir(self, target_dir, batch_size, epochs,one_hot=True):
        it = self.recalliterator(target_dir,
                                 batch_size=batch_size,
                                 epochs=epochs,  # 1400//10 = 140, 140*3 = 420 total call
                                 return_label=True)

        target_list = self.glob_dir(target_dir)  # amount 計算用
        steps = self.amount // batch_size


        for i in range(self.total_call):
            batch_flist, batch_label = next(it)
            # print(i, "|", over_batch)
            # print("  batch num: ", len(over_batch) )

            batch_data = self.loadImageFromList(batch_flist, 224)
            # print("  batch data: ", batch_data.shape)

            # convert to np.ndarray
            batch_data = np.array(batch_data)
            batch_label = np.array(batch_label)

            if one_hot:
                batch_label = np.identity(len(self.class_list))[batch_label.astype(np.int8)]

            yield batch_data, batch_label



if __name__ == "__main__":

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)
    data_dir = os.path.join(prj_root, "datasets")
    data_src = os.path.join(data_dir, "small_721")

    myimgh = MyImageDataHandler()

    # purpose_list = os.listdir(data_src)
    # purpose_list = myimgh.dirsieve(purpose_list)
    purpose_list = ['train', 'validation', 'test']

    train_dir = os.path.join(data_src, purpose_list[0])
    class_list = os.listdir(train_dir)
    class_list = myimgh.dirsieve(class_list)


    train_dir = os.path.join(data_src, "train")

    target_list = myimgh.glob_dir(train_dir)  # amount 計算用
    amount = len(target_list)
    # print(target_list)
    # print(amount)

    set_epochs = 3
    batch_size = 10
    steps = amount // batch_size

    print("\ntest glob_dir():")
    target_list, labels = myimgh.glob_dir(train_dir, return_label=True)  # amount 計算用
    amount = len(target_list)
    label_amount = len(labels)
    # print(target_list)
    print(amount)
    print(labels.shape)

    # ここから iterator 関数の check
    print("\ntest diriterator():")
    list_it = myimgh.diriterator(train_dir,
                                 batch_size=batch_size,
                                 return_label=True)
    for i in range(80):
       print(next(list_it))
    # batch_list, labels = myimgh.diriterator(train_dir,

    print("\ntest recalliterator():")
    list_it = myimgh.recalliterator(train_dir,
                                       batch_size=batch_size,
                                       epochs=set_epochs,
                                       return_label=True)
    for i in range(steps*set_epochs):
       print(next(list_it))
    # batch_list, labels = myimgh.diriterator(train_dir,

    print("\ntest flowfromdir():")
    ffd_it = myimgh.flowfromdir(train_dir,
                                batch_size=batch_size,
                                epochs=set_epochs,
                                one_hot=True)
    for i in range(75):  # (steps*set_epochs):
        batch_data, batch_label = next(ffd_it)
        print("  batch data: ", batch_data.shape)
        print("  batch label: ", batch_label)

        for i in range(len(batch_data)):
            plt.subplot(2, 5, i+1)
            plt.imshow(batch_data[i])
            plt.axis(False)
            plt.title(batch_label[i])
        plt.show()


    # batch_list, labels = myimgh.diriterator(train_dir,


    """
    # it = diriterator(train_dir, batch_size)
    reit = myimgh.recalliterator(train_dir,
                                 batch_size=batch_size,
                                 # epochs=set_epochs)  # 700//10 = 70, 70*3 = 210 total call
                                 epochs=set_epochs)  # 1400//10 = 140, 140*3 = 420 total call
    start = time.time()
    for i in range(steps*set_epochs):
        over_batch = next(reit)
        print(i, "|", over_batch)
        print("  batch num: ", len(over_batch) )

        batch_data = myimgh.loadImageFromList(over_batch, 224)
        print("  batch data: ", batch_data.shape)
    print("elapsed time: ", time.time() - start, " [sec]")
    """


    """
    # test origin img_utils utility -----
    print("\ntesting load_img():")
    sample_cat = os.path.join(train_dir, target_list[0])
    single_img_array = myimgh.load_img(sample_cat, 224)
    print("  result: ", single_img_array.shape)

    print("\ntesting loadImageFrom`List`():")
    data = myimgh.loadImageFromList(target_list, 224)
    print("  result: ", data.shape)

    
    print("\ntesting loadImageFromDir():")
    train_cat_dir = os.path.join(train_dir, "cat")
    train_cat_datas = myimgh.loadImageFromDir(train_cat_dir, 224)
    print("  result: ", train_cat_datas.shape)


    print("\ntesting inputDataCreator(train_dir, 224, normalize=False, one_hot=True):")
    data, label = myimgh.inputDataCreator(train_dir,
                                          224,
                                          normalize=False,
                                          one_hot=True)
    print("  result (data shape) : ", data.shape)
    # print("    data: \n", data[0])
    print("  result (label shape): ", label.shape)
    # print("    label: \n", label)

    print("\ntesting datasplit() <split rate => 7:2:1> :")
    data, label = myimgh.inputDataCreator(train_dir,
                                          224,
                                          normalize=True,
                                          one_hot=True)
    print(data.shape)
    print(label.shape)

    splited_data = myimgh.dataSplit(data, label)
    train_data, train_label = splited_data[0], splited_data[1]
    validation_data, validation_label  = splited_data[2], splited_data[3]
    test_data, test_label = splited_data[4], splited_data[5]
    print(train_label.shape)
    print(train_label[0])
    """

    # ----------

    """
    parser = argparse.ArgumentParser(description="画像読み込みに関する自家製ミニマルライブラリ (速度はあまりコミットしてないです..)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--display", type=int, default=99999)

    args = parser.parse_args()

    if args.test:


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

    if args.split:

    print("Done.")
    """
