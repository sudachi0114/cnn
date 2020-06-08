
import os, shutil

class Datasetmaker:

    def __init__(self):

        self.dirs = {}
        self.dirs['cwd'] = os.getcwd()
        self.dirs['datasets_dir'] = os.path.dirname(self.dirs['cwd'])
        # self.dirs['origin_dir'] = os.path.join(self.dirs['datasets_dir'], "origin")
        # self.data_amount = len(os.listdir(self.dirs['origin_data_dir']))  # origin はクラスごと別れていない
        self.dirs['cdev_origin_dir'] = os.path.join(self.dirs['datasets_dir'],
                                                    "cdev_origin")  # クラスごとに分離したディレクトリ

        # train / valid / test data を配置する dir
        self.division_list = ["train", "validation", "test"]

        # class label name
        self.cls_list = ["cat", "dog"]

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



    def dlist_sieve(self, DLIST):
        """ remove systemfiles utility
            # Args:
                DLIST (list): list of files
            # Returns:
                DLIST (list): sieved and sorted list
        """

        ignore_list = [".DS_Store", "__pycache__"]

        for igfile in ignore_list:
            if igfile in DLIST:
                DLIST.remove(igfile)

        return sorted(DLIST)


    # def separate(self, split_size='small_721', save_dir=None, begin_idx=0):
    def separate(self, AMOUNT, SEP_RATE, SAVE_DIR=None, begin_idx=0):

        # separate したデータの保存先を作成
        if SAVE_DIR is None:
            SAVE_DIR = os.path.join(self.dirs['datasets_dir'],
                                    "sample")
            print("SAVE_DIR : ", SAVE_DIR)
            os.makedirs(SAVE_DIR, exist_ok=True)
        elif type(SAVE_DIR) == str :
            SAVE_DIR = SAVE_DIR


        """
        if type(split_size) == str:
            train_size = self.split_dict[split_size][0]
            validation_size = self.split_dict[split_size][1]
            test_size = self.split_dict[split_size][2]
        elif type(split_size) == list:
            assert len(split_size) == 3
            train_size = split_size[0]
            validation_size = split_size[1]
            test_size = split_size[2]
        """

        cls_amount = AMOUNT // 2
        train_size = int(cls_amount*SEP_RATE['train'])
        validation_size = int(cls_amount*SEP_RATE['validation'])
        test_size = int(cls_amount*SEP_RATE['test'])


        # index ----------------------------------------
        # train     : 0 (or bagin_idx) ~ amount*train_rate = train_end
        # validation: train_end + amount*validation_rate = validation_end
        # test      : valdaiton_end  + test_size
        if begin_idx == 0:
            train_begin = 0
            train_end = train_size
        elif type(begin_idx) == int:
            train_begin = begin_idx
            train_end = begin_idx + train_size
            
        validation_begin = train_end
        validation_end = validation_begin + validation_size
        test_begin = validation_end
        test_end = validation_end + test_size

        print('-*-'*10)


        for division in self.division_list:
            div_dir = os.path.join(SAVE_DIR, division)
            print("division dir : ", div_dir)
            os.makedirs(div_dir, exist_ok=True)

            for cname in self.cls_list:
                # division/cname
                div_cls_dir = os.path.join(div_dir, cname)
                print("division class dir :", div_cls_dir)
                os.makedirs(div_cls_dir, exist_ok=True)

                pic_name_list = []
                if division == "train":
                    begin = train_begin
                    end = train_end
                    size = train_size

                elif division == "validation":
                    begin = validation_begin
                    end = validation_end
                    size = validation_size

                elif division == "test":
                    begin = test_begin
                    end = test_end
                    size = test_size
                    

                print("Amount of {}/{} pictures is : {}".format(division, cname, size))
                print("{} range is {} ~ {}".format(division, begin, end))
                for i in range(begin, end):
                    pic_name_list.append("{}.{}.jpg".format(cname, i))

                print("Copy name : {}/{} | pic_name_list : {}".format(division, cname, pic_name_list))

                assert len(pic_name_list) == size

                for pic_name in pic_name_list:
                    copy_src = os.path.join(self.dirs['cdev_origin_dir'], cname, pic_name)
                    copy_dst = os.path.join(div_cls_dir, pic_name)
                    shutil.copy(copy_src, copy_dst)

                print("Done.")

            print('-*-'*10)

    # def makeMul(self, N, AMOUNT, SEP_RATE):
    def makeMul(self, N, AMOUNT, SEP_RATE):
        cdev_cls_dir = os.path.join(self.dirs['cdev_origin_dir'], self.cls_list[0])
        cls_all_amount = os.listdir(cdev_cls_dir)
        cls_all_amount = len( self.dlist_sieve(cls_all_amount) )

        max_N = cls_all_amount // AMOUNT

        if N > max_N:
            raise Exception('To Large N (overflow)')
        else:
            msample_dir = os.path.join(self.dirs['datasets_dir'], "mulSample", "m_sample")
            os.makedirs(msample_dir, exist_ok=True)

            for i in range(N):
                begin_idx = i*AMOUNT

                save_dir = os.path.join(msample_dir, "sample_{}".format(i))
                self.separate(AMOUNT, SEP_RATE, SAVE_DIR=save_dir, begin_idx=begin_idx)

        


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

    ins = Datasetmaker()

    sep_rate = {"train":0.7, "validation":0.2, "test":0.1}
    ins.separate(AMOUNT=1000, SEP_RATE=sep_rate)
    # ins.separate(AMOUNT=25000, SEP_RATE=sep_rate)
    # ins.makeMul(N=3, AMOUNT=100, SEP_RATE=sep_rate)

    """
    if args.make_dataset:
        ds.separate(split_size='small_721')

    if args.make_gtest:
        ds.makeGlobalTest()
    """

