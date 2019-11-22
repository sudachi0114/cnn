
import sys, os, shutil
sys.path.append(os.pardir)

from utils.img_utils import inputDataCreator
from utils.aug_with_imgaug import AugWithImgaug

# 様々な変換を施したデータを生成
auger = AugWithImgaug()

cwd = os.getcwd()
prj_root = os.path.dirname(cwd)

class_list = ["cat", "dog"]


def severalCreate(aug, num=3):

    print("Create Augmented Pictures several times!")

    target_dir = os.path.join(prj_root, "dogs_vs_cats_smaller", "train")

    for i in range(num):
        print("\nprocess { ", aug, " | ", i,  " times } .....")
        auger.save_imgauged_img(target_dir,
                                224,
                                normalize=False,
                                save_dir=cwd,
                                aug=aug)
        print("  <= rename.")
        os.rename("./dogs_vs_cats_auged_{}".format(aug),
                  "./dogs_vs_cats_auged_{}_{}".format(aug, i))
        print("Done.")


    print("lastly creating { native } ...")
    auger.save_imgauged_img(target_dir,
                            224,
                            normalize=False,
                            save_dir=cwd,
                            aug="native")
    print("Done.")





def severalConcat(aug, num):

    # 対応するディレクトリを作成
    concat_data_dir = os.path.join(cwd, "da_concat")
    os.makedirs(concat_data_dir, exist_ok=True)


    # select されたものの中から
    for i in range(num+1):
        pic_list = []
        if i == 3:
            cp_source_folder = "dogs_vs_cats_auged_native"

            for class_name in class_list:  # 各クラスにアクセス
                concat_target = os.path.join(cp_source_folder, class_name)  # native/cat
                pic_list = os.listdir(concat_target)  # cp_source_folder にある写真のリスト
                print("\n以下のファイルを移動します:\n", pic_list)
                print("    個数: ", len(pic_list))

                cp_dist_dir = os.path.join(concat_data_dir, class_name)  # 管理番号/cat などを作成
                os.makedirs(cp_dist_dir, exist_ok=True)

                for pic_name in pic_list:
                    copy_src = os.path.join(concat_target, pic_name)  # ココにある写真を
                    copy_dst = os.path.join(cp_dist_dir, pic_name)  # こっちにコピー
                    shutil.copy(copy_src, copy_dst)

        else:
            cp_source_folder = "dogs_vs_cats_auged_{}_{}".format(aug, i)  # 各フォルダの名前を作って

            for class_name in class_list:  # 各クラスにアクセス
                concat_target = os.path.join(cp_source_folder, class_name)  # ここがターゲット dogs_vs_cats_native/cat など
                pic_list = os.listdir(concat_target)  # cp_source_folder にある写真のリストを取得
                print("\n以下のファイルを移動します:\n", pic_list)
                print("    個数: ", len(pic_list))

                # 中身の名前を変更
                rename_pic_list = []
                for f in pic_list:
                    before = os.path.join(concat_target, f)
                    after = os.path.join(concat_target, "{}_{}".format(i, f))
                    os.rename(before, after)

                    # pic_list を更新
                    rename_pic_list = os.listdir(concat_target)

                cp_dist_dir = os.path.join(concat_data_dir, class_name)  # 管理番号/cat などを作成
                os.makedirs(cp_dist_dir, exist_ok=True)

                for pic_name in rename_pic_list:
                    copy_src = os.path.join(concat_target, pic_name)  # ココにある写真を
                    copy_dst = os.path.join(cp_dist_dir, pic_name)  # こっちにコピー
                    shutil.copy(copy_src, copy_dst)

    


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

    parser.add_argument("--several",
                        nargs=2,
                        help="create augment pics several times")

    parser.add_argument("--several_concat",
                        nargs='*',
                        help="concat augmented data define by your command line args.")
    
    parser.add_argument("--clean",
                        action="store_true",
                        help="remove auged datas.")

    args = parser.parse_args()
    
    if args.several:
        print(args.several)
        arg = args.several[0]
        num = int(args.several[1])
        severalCreate(arg, num)

    if args.several_concat:
        print(args.several_concat)
        aug = args.several_concat[0]
        num = int(args.several_concat[1])
        severalConcat(aug, num)


    if args.clean:
        clean()
