
import sys, os, shutil
sys.path.append(os.pardir)

from utils.img_utils import inputDataCreator
from utils.aug_with_imgaug import AugWithImgaug

# 様々な変換を施したデータを生成
auger = AugWithImgaug()

cwd = os.getcwd()
prj_root = os.path.dirname(cwd)


picked_aug_list = ["rotation", "hflip", "gnoise", "invert", "native"]
# native を最後に置く事で、管理番号が奇数なら native が含まれている保証をできるようにする。

class_list = ["cat", "dog"]


def CreateDaPictures(aug):

    print("Create Augmented Pictures!")

    target_dir = os.path.join(prj_root, "dogs_vs_cats_smaller", "train")

    if type(aug) == str:
        auger.save_imgauged_img(target_dir,
                                224,
                                normalize=False,
                                save_dir=cwd,
                                aug=aug)
        print("Done.")


    elif type(aug) == list:
        for picked_aug in aug:
            print("\nprocess { ", picked_aug, " } .....")
            auger.save_imgauged_img(target_dir,
                                    224,
                                    normalize=False,
                                    save_dir=cwd,
                                    aug=picked_aug)
            print("Done.")


def SieveDirList():

    print("checking directory .....")

    raw_cwd_list = os.listdir(cwd)
    # print("raw: ", raw_cwd_list)

    auged_data_dirs = []
    for canditate in raw_cwd_list:
        # print("canditate: ", canditate)
        if "dogs_vs_cats_auged" in canditate:
            auged_data_dirs.append(canditate)

    return auged_data_dirs


def someConcat(target_list):

    # 対応するディレクトリを作成
    concat_data_dir = os.path.join(cwd, "da_concat_some_{}".format(len(target_list)))
    os.makedirs(concat_data_dir, exist_ok=True)


    # select されたものの中から
    for aug_name in target_list:
        cp_source_folder = "dogs_vs_cats_auged_{}".format(aug_name)  # 各フォルダの名前を作って

        for class_name in class_list:  # 各クラスにアクセス
            concat_target = os.path.join(cp_source_folder, class_name)  # ここがターゲット dogs_vs_cats_native/cat など
            pic_list = os.listdir(concat_target)  # cp_source_folder にある写真のリストを取得
            print("\n以下のファイルを移動します:\n", pic_list)
            print("    個数: ", len(pic_list))

            cp_dist_dir = os.path.join(concat_data_dir, class_name)  # 管理番号/cat などを作成
            os.makedirs(cp_dist_dir, exist_ok=True)

            for pic_name in pic_list:
                copy_src = os.path.join(concat_target, pic_name)  # ココにある写真を
                copy_dst = os.path.join(cp_dist_dir, pic_name)  # こっちにコピー
                shutil.copy(copy_src, copy_dst)



def daPicsConcat(kanri_no):

    # 組み合わせの "あるなし" を 2進法で取得
    bin_kanri_no = format(kanri_no, '05b')  # 5桁 まで 0 埋め
    print("\n[ ", kanri_no, " : ", bin_kanri_no, " ]")

    # 対応するディレクトリを作成
    concat_data_dir = os.path.join(cwd, "da_concat_{}".format(kanri_no))
    os.makedirs(concat_data_dir, exist_ok=True)

    selected_auged = []
    for j in range(len(bin_kanri_no)):
        if bin_kanri_no[j] == "1":  # flg が立っている場合は
            selected_auged.append(picked_aug_list[j])  # そのDAパターンを選択
            # print("    select: ", picked_aug_list[j])


    # select されたものの中から
    for aug_name in selected_auged:
        cp_source_folder = "dogs_vs_cats_auged_{}".format(aug_name)  # 各フォルダの名前を作って

        for class_name in class_list:  # 各クラスにアクセス
            concat_target = os.path.join(cp_source_folder, class_name)  # ここがターゲット dogs_vs_cats_native/cat など
            pic_list = os.listdir(concat_target)  # cp_source_folder にある写真のリストを取得
            print("\n以下のファイルを移動します:\n", pic_list)
            print("    個数: ", len(pic_list))

            cp_dist_dir = os.path.join(concat_data_dir, class_name)  # 管理番号/cat などを作成
            os.makedirs(cp_dist_dir, exist_ok=True)

            for pic_name in pic_list:
                copy_src = os.path.join(concat_target, pic_name)  # ココにある写真を
                copy_dst = os.path.join(cp_dist_dir, pic_name)  # こっちにコピー
                shutil.copy(copy_src, copy_dst)


def concatWhole():

    auged_data_dirs = SieveDirList()

    if auged_data_dirs == []:
        CreateDaPictures(picked_aug_list)
        auged_data_dirs = SieveDirList()

    for i in range(2**len(auged_data_dirs)):
        daPicsConcat(i)

        


def clean():

    cwd_list = os.listdir(cwd)

    rm_file_canditate = []
    for file_name in cwd_list:
        if "dogs_vs_cats_auged" in file_name:
            rm_file_canditate.append(file_name)
        elif "da_concat" in file_name:
            rm_file_canditate.append(file_name)

    for file_name in rm_file_canditate:
        print("-> find: ", file_name)

    ans = input("!! これらのファイルを削除してもいいですか? (yes:y / no:n) >>> ")
    if ans == 'y':
        for file_name in rm_file_canditate:
            shutil.rmtree(file_name)
    else:
        pass



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Augment を 4種類行い、そのデータを保存するプログラム")

    parser.add_argument("--make",
                        action="store_true",
                        help="augmented data creation & concat >>whole<<.")
    parser.add_argument("--create_from_list",
                        action="store_true",
                        help="augmented data creation by picked_aug_list = [rotation, hflip, gnoise, invert, native].")
    parser.add_argument("--create_pics",
                        nargs='*',
                        help="augmented data creation define by your command line args.")
    parser.add_argument("--some_concat",
                        nargs='*',
                        help="concat augmented data define by your command line args.")
    parser.add_argument("--create",
                        default=32,
                        type=int,
                        help="concat augmented data.")
    parser.add_argument("--clean",
                        action="store_true",
                        help="remove auged datas.")

    args = parser.parse_args()
    
    if args.create_pics:
        CreateDaPictures(args.create_pics)
    if args.create_from_list:
        CreateDaPictures(picked_aug_list)

    if args.some_concat:
        someConcat(args.some_concat)
    if args.create < 32:
        daPicsConcat(args.create)

    if args.make:
        concatWhole()

    if args.clean:
        clean()
