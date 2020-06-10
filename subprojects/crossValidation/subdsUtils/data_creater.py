import os
import sys
import shutil
sys.path.append(os.pardir)

from PIL import Image
from dsUtils.imgaug_auger import AugWithImgaug


class DatasetMaker:

    def __init__(self):

        self.dirs = {}

        self.dirs["cwd"] = os.getcwd()
        self.dirs["sub_prjr"] = os.path.dirname(self.dirs["cwd"])
        self.dirs["sub_datasetsd"] = os.path.join(self.dirs["sub_prjr"], "subdatasets")

        self.dirs["prj_root"] = os.path.dirname(os.path.dirname(self.dirs["sub_prjr"]))
        self.origin_loc = os.path.join(self.dirs["prj_root"], "datasets", "origin")

        self.cls_list = ['cat', 'dog']

        self.amount = len(os.listdir(self.origin_loc))


        # define
        # train_size(rate) = 0.7
        # validation_size = 0.2
        # test_size = 0.1
        # sep_rate = {"train":0.7, "validation":0.2, "test":0.1}



    def separete(self, N=None, DATASET_SIZE=1000, SAVE_DIR=None):

        if N is None:
            sample_num = self.amount // DATASET_SIZE
            # 25000 / 1000 => 25 sample
        else:
            sample_num = N

        print("total: ", self.amount)
        print("dataset size: ", DATASET_SIZE)
        print("num of sample: ", sample_num)


        for i in range(sample_num):
            if SAVE_DIR is None:
                save_loc = os.path.join(self.dirs["sub_datasetsd"],
                                        "sample_{}".format(i),
                                        "natural")
                os.makedirs(save_loc, exist_ok=False)
            else:
                save_loc = SAVE_DIR

            each_cls_dataset_size = DATASET_SIZE // 2 
            print("  each class's data size: ", each_cls_dataset_size)

            idx_start = i * each_cls_dataset_size
            idx_end = idx_start + each_cls_dataset_size

            # Cross Validation を行うので division には分けない
            for cname in self.cls_list:
                each_cls_save_loc = os.path.join(save_loc, cname)
                os.makedirs(each_cls_save_loc, exist_ok=True)
                print("\nmake directory: ", each_cls_save_loc)

                # copy
                pic_name_list = []
                print("  Amount of {} pictures is : {}".format(cname, each_cls_dataset_size))
                print("  data range is from {} to {}".format(idx_start, idx_end))

                for i in range(idx_start, idx_end):
                    pic_name_list.append("{}.{}.jpg".format(cname, i))

                print("    !! check squence: ")
                assert len(pic_name_list) == each_cls_dataset_size
                print("    !!  -> cleared.")

                for pic_name in pic_name_list:
                    copy_src = os.path.join(self.origin_loc, pic_name)
                    copy_dst = os.path.join(each_cls_save_loc, pic_name)
                    shutil.copy(copy_src, copy_dst)
                print("Collectly Copied.")

            print('-*-'*10)


    def clean(self):

        cand_list = os.listdir(self.dirs["sub_datasetsd"])

        found = []
        for elem in cand_list:
            if "sample_" in elem:
                found.append(elem)

        if len(found) == 0:
            print("削除対象のファイルはみつかりませんでした。")
        else:
            print("found {} items below -----".format(len(found)))
            for item in found:
                print("-> ", item)

            exec_rm = input("\nこれらのフォルダを消去しますか? (yes: y / no: n) >>> ")
            if exec_rm == 'y':
                for item in found:
                    rm_targ = os.path.join(self.dirs["sub_datasetsd"], item)
                    shutil.rmtree(rm_targ)
                print("   削除しました。")
            else:
                print("    削除を中止しました。")


    def augment(self, TARGET_DIR, AUGMENTATION="rotation"):

        # target_dir = os.path.join(TARGET_DIR, "natural")

        target_dir = TARGET_DIR
        print("Augment {} datas...".format(target_dir))
        assert len(os.listdir(target_dir)) > 0


        print("Augment mode: ", AUGMENTATION)

        #train_data_location = os.path.join(TARGET_DIR, "train")

        auger = AugWithImgaug()
        data, label = auger.imgaug_augment(
            TARGET_DIR=os.path.join(TARGET_DIR, "natural"),
            INPUT_SIZE=224,
            NORMALIZE=False,
            AUGMENTATION=AUGMENTATION)

        print("data shape: ", data.shape)
        print("label shape: ", label.shape)

        save_data_shape = data[0].shape

        #data *= 255

        for j, cname in enumerate(self.cls_list):
            print("\nsave {} class after generation".format(cname))
            idx = 0  # int(amount / 2)
            for i, each_data in enumerate(data):
                if label[i] == j:
                    auged_cls_data_loc = os.path.join(target_dir,
                                                      "auged", cname)
                    os.makedirs(auged_cls_data_loc, exist_ok=True)
                    save_img_path = os.path.join(auged_cls_data_loc,
                                                 "{}.{}.{}.jpg".format(cname, AUGMENTATION, idx))

                    assert each_data.shape == save_data_shape
                    pil_auged_img = Image.fromarray(each_data.astype('uint8'))  # float の場合は [0,1]/uintの場合は[0,255]で保存
                    pil_auged_img.save(save_img_path)
                    idx += 1
            print("Done.")
        print("Collectly Saved.")



    def r_augment(self, TARGET_DIR=None, AUGMENTATION="rotation"):

        targets = os.listdir(self.dirs["sub_datasetsd"])
        for fold in targets:
            target = os.path.join(self.dirs["sub_datasetsd"],
                                  fold)
            print("process {}".format(fold))
            self.augment(TARGET_DIR=target,
                         AUGMENTATION=AUGMENTATION)



    def concat(self, TARGET_DIR):

        for cname in self.cls_list:
            each_concat_cls_data_loc = os.path.join(TARGET_DIR,
                                                    "concat", cname)
            os.makedirs(each_concat_cls_data_loc, exist_ok=True)
            print("\nmake directory: ", each_concat_cls_data_loc)

            each_cls_natural_data = os.path.join(TARGET_DIR, "natural", cname)
            each_cls_auged_data = os.path.join(TARGET_DIR, "auged", cname)


            copy_list = []

            for nat_img in os.listdir(each_cls_natural_data):
                copy_list.append( os.path.join(each_cls_natural_data, nat_img) )
            for auged_img in os.listdir(each_cls_auged_data):
                copy_list.append( os.path.join(each_cls_auged_data, auged_img) )

            print(copy_list, len(copy_list))
            assert len(copy_list) == 2*len(os.listdir(each_cls_natural_data))

            

            for pic_loc in copy_list:
                copy_src = pic_loc
                copy_dst = os.path.join(each_concat_cls_data_loc)
                shutil.copy(copy_src, copy_dst)
            print("Collectly Concated.")

            print("\n----------\n")



    def r_concat(self, TARGET_DIR=None):

        targets = os.listdir(self.dirs["sub_datasetsd"])
        for fold in targets:
            target = os.path.join(self.dirs["sub_datasetsd"],
                                  fold)
            print("process {}".format(fold))
            self.concat(TARGET_DIR=target)



    def doWhole(self, N, DATASET_SIZE, AUGMENTATION="rotation"):

        self.separete(N, DATASET_SIZE)
        self.r_augment(AUGMENTATION=AUGMENTATION)
        self.r_concat()





if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Data Augmentation を 100例で試そう (データ用意プログラム編)")

    parser.add_argument("--separete", action="store_true", help="ディレクトリとデータを作成します。")
    parser.add_argument("--augment", action="store_true", help="分割したデータに Data Augmentation を施します。(!! PATH を指定 !!)")
    parser.add_argument("--raugment", action="store_true", help="分割した全てのデータに Data Augmentation を施します。")
    parser.add_argument("--concat", action="store_true", help="DA したデータとナチュラルなデータを結合。")
    parser.add_argument("--rconcat", action="store_true", help="DA したデータとナチュラルなデータを全てのサンプルに対して結合。")
    parser.add_argument("--make", action="store_true", help="ディレクトリとデータを作成し水増しを行い、これら 2つのデータを結合します。")
    parser.add_argument("--clean", action="store_true", help="作成したディレクトリを削除します。")

    arg = parser.parse_args()

    ins = DatasetMaker()
    print(ins.__dict__)

    if arg.separete:
        # ins.separete()
        ins.separete(N=5, DATASET_SIZE=500)
    if arg.augment:
        ins.augment(os.path.join("../subdatasets/sample_0"))
    if arg.raugment:
        ins.r_augment()
    elif arg.concat:
        ins.concat(os.path.join("../subdatasets/sample_0"))
    elif arg.rconcat:
        ins.r_concat()
    elif arg.make:
        ins.doWhole(N=5,
                    DATASET_SIZE=100,
                    AUGMENTATION="rotation")
    elif arg.clean:
        ins.clean()
