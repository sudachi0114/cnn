
import os, sys
sys.path.append(os.pardir)

import shutil
from PIL import Image

from separator import DataSeparator
from imgaug_auger import AugWithImgaug
# from utils.aug_with_imgaug import AugWithImgaug

# define ----------
cwd = os.getcwd()
sub_prj = os.path.dirname(cwd)
print(sub_prj)

origin_dir = os.path.join(cwd, "origin")

purpose_list = ['train', 'validation', 'test']
class_list = ['cat', 'dog']

# クラス合算で得られるので 2 で割って, class 毎に変更
amount = len(os.listdir(origin_dir)) // 2

# class 毎 (総量は 2倍)
split_list = [600, 300, 100]

total_size = sum(split_list)
sample_size = amount // total_size

print(total_size, sample_size)



def sampling():

    ds = DataSeparator()

    for i in range(sample_size):
        if i == 0:
            train_begin = 0
        else:
            train_begin = test_end

        # save_location = os.path.join(cwd, "sample{}".format(i))
        # os.makedirs(save_location, exist_ok=True)

        ds.separate(split_size=split_list,
                    save_dir="sample{}".format(i),
                    begin_idx=train_begin)
        test_end = train_begin + total_size


def augment():

    selected_mode = "rotation"
    print("Augment mode: ", selected_mode)

    auger = AugWithImgaug()

    """
    for i in range(sample_size):
        target_dir = os.path.join(cwd,
                                  "sample{}".format(i),
                                  "train")
        print("process", target_dir)
        auger.save_imgauged_img(target_dir,
                                224,
                                normalize=False,
                                aug=selected_mode)
        print("Done.")
    """

    """ 複数回 変換を行う場合
    """
    for i in range(sample_size):
        for j in range(2):
            print("process {} times".format(j+1))
            target_dir = os.path.join(cwd,
                                      "sample{}".format(i),
                                      "train")
            print("process", target_dir)

            pparent, origin_dir = os.path.split(target_dir)
            save_dir = selected_mode + "_" + origin_dir + "_{}".format(j)
            save_loc = os.path.join(pparent, save_dir)
            print(save_dir)

            auger.save_imgauged_img(target_dir,
                                    224,
                                    save_dir=save_loc,
                                    normalize=False,
                                    aug=selected_mode)
            print("Done.")
    print("All Sample Collectly Saved.")




def clean():

    cwd_list = sorted( os.listdir(cwd) )

    found = []
    for elem in cwd_list:
        if "sample" in elem:
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
                shutil.rmtree(item)
            print("   削除しました。")
        else:
            print("    削除を中止しました。")



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Data Augmentation を 100例で試そう (データ用意プログラム編)")

    parser.add_argument("--separete", action="store_true", help="ディレクトリとデータを作成します。")
    parser.add_argument("--augment", action="store_true", help="分割下データに Data Augmentation を施します。")
    parser.add_argument("--concat", action="store_true", help="DA したデータと original データを結合。")
    parser.add_argument("--make", action="store_true", help="ディレクトリとデータを作成し水増しを行い、これら 2つのデータを結合します。")
    parser.add_argument("--clean", action="store_true", help="作成したディレクトリを削除します。")

    arg = parser.parse_args()

    if arg.separete:
        sampling()
    if arg.augment:
        augment()  # os.path.join(cwd, "experiment_0")
    elif arg.make:
        sampling()
        augment()
    elif arg.clean:
        clean()
