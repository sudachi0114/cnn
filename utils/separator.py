
import os, shutil

class DataSeparator:

    def __init__(self):

        self.dirs = {}
        self.dirs['cnn_dir'] = os.path.dirname(os.getcwd())
        self.dirs['origin_data_dir'] = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_origin")
        self.data_amount = len(os.listdir(self.dirs['origin_data_dir']))  # クラスごと別れていない前提

        # train / valid / test data を配置する dir
        self.data_purpose_list = ["train", "validation", "test"]

        # class label name
        self.class_label = ["cat", "dog"]

        # dict = { split_size:[train_num, validation_num, test_num] }
        self.split_dict = {}
        self.split_dict['smaller'] = [50, 25, 25]
        self.split_dict['mid300'] = [150, 50, 50]
        self.split_dict['full'] = [10000, 2500, 2500]
        self.split_dict['mid_test'] = [50, 25, 50]
        self.split_dict['large_test'] = [50, 25, 100]



    def separate(self, split_size='smaller'):

        self.split_size = split_size

        # separate したデータの保存先を作成
        self.dirs['save_dir'] = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_{}".format(self.split_size))
        print("save_dir : ", self.dirs['save_dir'])
        os.makedirs(self.dirs['save_dir'], exist_ok=True)


        train_num = self.split_dict[self.split_size][0]
        validation_num = self.split_dict[self.split_size][1]
        test_num = self.split_dict[self.split_size][2]

        validation_begin = train_num
        validation_end = train_num + validation_num

        test_begin = validation_end
        test_end = validation_end + test_num


        print('-*-'*10)


        for name in self.data_purpose_list:
            print("make directry : {}..".format(name))
            target_dir = os.path.join(self.dirs['save_dir'], name)
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
                    print("Amount of {}/{} pictures is : {}".format(name, class_name, train_num) )
                    print("train range is {} to {}".format(0, train_num))
                    for i in range(train_num):  # (0 ~ 49) の50枚分
                        pic_name_list.append("{}.{}.jpg".format(class_name, i))

                elif name == "validation":
                    print("Amount of {}/{} pictures is : {}".format(name, class_name, validation_num) )
                    print("validation range is {} to {}".format(validation_begin, validation_end))
                    for i in range(validation_begin, validation_end):  # (50 ~ 74) の 25枚分
                        pic_name_list.append("{}.{}.jpg".format(class_name, i))

                elif name == "test":
                    print("Amount of {}/{} pictures is : {}".format(name, class_name, test_num))
                    print("test range is {} to {}".format(test_begin, test_end))
                    for i in range(test_begin, test_end):  # (75 ~ 99) の 25枚分
                        pic_name_list.append("{}.{}.jpg".format(class_name, i))

                print("Copy name : {}/{} | pic_name_list : {}".format(name, class_name, pic_name_list))

                for pic_name in pic_name_list:
                    copy_src = os.path.join(self.dirs['origin_data_dir'], pic_name)
                    copy_dst = os.path.join(target_class_dir, pic_name)
                    shutil.copy(copy_src, copy_dst)

                print("Done.")

            print('-*-'*10)

    def makeGlobalTest(self):

        # global_test data の保存先を作成
        self.dirs['save_dir'] = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_global_test")
        print("save_dir : ", self.dirs['save_dir'])
        # os.makedirs(self.dirs['save_dir'], exist_ok=True)

        idx_end = self.data_amount / 2

        print(idx_end)




if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="origin data から トレーニング用のデータを切り分けるプログラム")

    parser.add_argument("--make_dataset", action="store_true", default=False, help="任意の大きさのデータセットを作成")
    parser.add_argument("--make_gtest", action="store_true", default=False, help="global test を作成")

    args = parser.parse_args()

    ds = DataSeparator()

    if args.make_dataset:
        ds.separate(split_size='smaller')
        #ds.separate(split_size='large_test')

    if args.make_gtest:
        ds.makeGlobalTest()

