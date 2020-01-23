
import os, shutil

class DataSeparator:

    def __init__(self):

        self.dirs = {}
        self.dirs['data_dir'] = os.getcwd()
        self.dirs['origin_data_dir'] = os.path.join(self.dirs['data_dir'], "origin")
        self.data_amount = len(os.listdir(self.dirs['origin_data_dir']))  # クラスごと別れていない前提

        # train / valid / test data を配置する dir
        self.data_purpose_list = ["train", "validation", "test"]

        # class label name
        self.class_label = ["cat", "dog"]

        # dict = { split_size:[train_size, validation_size, test_size] }
        self.split_dict = {}
        # self.split_dict['small'] = [50, 25, 25]
        # small : means amount = 1000
        # medium: means amount = 2000
        # large : means amount = 3000
        self.split_dict['small_721'] = [700, 200, 100]
        self.split_dict['small_631'] = [600, 300, 100]
        self.split_dict['medium_721'] = [1400, 400, 200]
        self.split_dict['medium_631'] = [1200, 600, 200]
        self.split_dict['large_721'] = [2100, 600, 300]
        self.split_dict['large_631'] = [1800, 900, 300]
        self.split_dict['full_721'] = [8750, 2500, 1250]
        self.split_dict['full_631'] = [7500, 3750, 1250]



    def separate(self, split_size='small_721'):

        # separate したデータの保存先を作成
        save_dir = os.path.join(self.dirs['data_dir'],
                                "{}".format(split_size))
        print("save_dir : ", save_dir)
        os.makedirs(save_dir, exist_ok=True)


        train_size = self.split_dict[split_size][0]
        validation_size = self.split_dict[split_size][1]
        test_size = self.split_dict[split_size][2]

        # index ----------------------------------------
        # train     : 0 ~ train_size-1
        # validation: train_size ~ train_size + validation_size
        # test      : valdaiton_end ~ validation_end + test_size
        validation_begin = train_size
        validation_end = validation_begin + validation_size
        test_begin = validation_end
        test_end = validation_end + test_size


        print('-*-'*10)


        for purpose in self.data_purpose_list:
            print("make directry : {}..".format(purpose))
            target_dir = os.path.join(save_dir, purpose)
            print("target_dir : ", target_dir)
            os.makedirs(target_dir, exist_ok=True)

            for cname in self.class_label:
                # train/cname
                print("make directry : {}/{}..".format(purpose, cname))
                target_class_dir = os.path.join(target_dir, cname)
                print("target_class_dir :", target_class_dir)
                os.makedirs(target_class_dir, exist_ok=True)

                pic_name_list = []
                if purpose == "train":
                    begin = 0
                    size = end = train_size   

                elif purpose == "validation":
                    begin = validation_begin
                    end = validation_end
                    size = validation_size

                elif purpose == "test":
                    begin = test_begin
                    end = test_end
                    size = test_size
                    

                print("Amount of {}/{} pictures is : {}".format(purpose, cname, size))
                print("{} range is {} ~ {}".format(purpose, begin, end))
                for i in range(begin, end):
                    pic_name_list.append("{}.{}.jpg".format(cname, i))

                print("Copy name : {}/{} | pic_name_list : {}".format(purpose, cname, pic_name_list))

                assert len(pic_name_list) == size

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
        os.makedirs(self.dirs['save_dir'], exist_ok=True)

        idx_end = int(self.data_amount / 2)


        test_size = 100  # / each_class
        
        begin = idx_end-test_size
        for class_name in self.class_label:
            pic_name_list = []
            # train/cats or dogs
            print("make directry : global_test/{}..".format(class_name))
            target_class_dir = os.path.join(self.dirs['save_dir'], class_name)
            print("target_class_dir :", target_class_dir)
            os.makedirs(target_class_dir, exist_ok=True)

            print("global test index range is from {} to {}".format(begin, idx_end))
            for i in range(begin, idx_end):
                pic_name_list.append("{}.{}.jpg".format(class_name, i))

            print("Copy name : gtest/{} | pic_name_list : {}".format(class_name, pic_name_list))

            assert len(pic_name_list) == test_size

            for pic_name in pic_name_list:
                copy_src = os.path.join(self.dirs['origin_data_dir'], pic_name)
                copy_dst = os.path.join(target_class_dir, pic_name)
                shutil.copy(copy_src, copy_dst)
            

            print("Done.")

        print('-*-'*10)
           
            




if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="origin data から" \
                                     "トレーニング用のデータを切り分けるプログラム")

    parser.add_argument("--make_dataset",
                        action="store_true",
                        default=False,
                        help="任意の大きさのデータセットを作成")
    parser.add_argument("--make_gtest",
                        action="store_true",
                        default=False,
                        help="global test を作成")

    args = parser.parse_args()

    ds = DataSeparator()

    if args.make_dataset:
        ds.separate(split_size='small_721')

    if args.make_gtest:
        ds.makeGlobalTest()

