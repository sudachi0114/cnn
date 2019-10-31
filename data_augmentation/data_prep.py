
import sys, os, shutil
sys.path.append(os.pardir)

from utils.data_separator import DataSeparator
from utils.da_handler import DaHandler

# ランダムにたくさん smaller サンプルを取るのはあとで行う。
#ds = DataSeparator()
#print(ds.__dict__)

# 様々な変換を施したデータを生成
dah = DaHandler()

cwd = os.getcwd()

# dogs_vs_cats_
picked_aug_list = ["rotation", "hflip", "gnoise", "invert", "native"]
# native を最後に置く事で、管理番号が奇数なら native が含まれている保証をできるようにする。

classes = ["cat", "dog"]


def daImgCreate():
    for picked_aug in picked_aug_list:
        print("\nprocess ", picked_aug, " .....")
        dah.save_imgauged_img(save_dir=cwd, save_mode='image', aug=picked_aug)
        print("Done.")


def daFolderConcatenate():

    raw_cwd_list = os.listdir(cwd)
    # print("raw: ", raw_cwd_list)

    auged_data_dirs = []
    for canditate in raw_cwd_list:
        # print("canditate: ", canditate)
        if "dogs_vs_cats_auged" in canditate:
            auged_data_dirs.append(canditate)


    for i in range(2**len(auged_data_dirs)):

        # 組み合わせの "あるなし" を 2進法で取得
        bin_kanri_no = format(i, '05b')  # 5桁 まで 0 埋め
        print("\n[ ", i, " : ", bin_kanri_no, " ]")

        # 対応するディレクトリを作成
        concat_data_dir = os.path.join(cwd, "da_concat_{}".format(i))
        os.makedirs(concat_data_dir, exist_ok=True)

        selected_auged = []
        for j in range(len(bin_kanri_no)):
            if bin_kanri_no[j] == "1":  # flg が立っている場合は
                selected_auged.append(picked_aug_list[j])  # そのDAパターンを選択
                # print("    select: ", picked_aug_list[j])


        # select されたものの中から
        for aug_name in selected_auged:
            cp_source_folder = "dogs_vs_cats_auged_{}".format(aug_name)  # 各フォルダの名前を作って
            for class_name in classes:  # 各クラスにアクセス
                concat_target = os.path.join(cp_source_folder, class_name)  # ここがターゲット dogs_vs_cats_native/cat など
                pic_list = os.listdir(concat_target)  # cp_source_folder にある写真のリストを取得
                print("以下のファイルを移動します:\n", pic_list)
                print("    個数: ", len(pic_list))

                cp_dist_dir = os.path.join(concat_data_dir, class_name)  # 管理番号/cat などを作成
                os.makedirs(cp_dist_dir, exist_ok=True)

                for pic_name in pic_list:
                    copy_src = os.path.join(concat_target, pic_name)  # ココにある写真を
                    copy_dst = os.path.join(cp_dist_dir, pic_name)  # こっちにコピー
                    shutil.copy(copy_src, copy_dst)






if __name__ == '__main__':

    # daImageCreate()
    daFolderConcatenate()




