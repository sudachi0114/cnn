import os, sys, shutil
sys.path.append(os.pardir)

from PIL import Image
from utils.da_handler import DaHandler

cwd = os.getcwd()
prj_root = os.path.dirname(cwd)

origin_data_location = os.path.join(prj_root, "dogs_vs_cats_origin")

purpose_list = ['train', 'validation', 'test']
class_list = ['cat', 'dog']

amount = len(os.listdir(origin_data_location))


def separete():

    train_size = 50
    validation_size = 50
    test_size = 100

    experiment_size = 5  #100

    for i in range(experiment_size):
        if i == 0:
            train_begin = 0

        else:
            train_begin = test_end

        train_end = train_begin + train_size

        validation_begin = train_end
        validation_end = validation_begin + validation_size

        test_begin = validation_end
        test_end = validation_end + test_size


        save_location = os.path.join(cwd, "experiment_{}".format(i))
        os.makedirs(save_location, exist_ok=True)
        for purpose in purpose_list:
            each_purpose_save_loc = os.path.join(save_location, purpose)
            os.makedirs(each_purpose_save_loc, exist_ok=True)
            for class_name in class_list:
                each_class_save_loc = os.path.join(each_purpose_save_loc, class_name)
                os.makedirs(each_class_save_loc, exist_ok=True)
                print("\nmake directory: ", each_class_save_loc)

                pic_name_list = []
                begin = 0
                end = 0
                size = 0

                if purpose == "train":
                    begin = train_begin
                    end = train_end
                    size = train_size

                elif purpose == "validation":
                    begin = validation_begin
                    end = validation_end
                    size = validation_size

                elif purpose == "test":
                    begin = test_begin
                    end = test_end
                    size = test_size

                print("  Amount of {}/{} pictures is : {}".format(purpose, class_name, size))
                print("  {} range is from {} to {}".format(purpose, begin, end))
                for i in range(begin, end):
                    pic_name_list.append("{}.{}.jpg".format(class_name, i))

                print("    !! check squence: ")
                assert len(pic_name_list) == size
                print("    !!  -> cleared.")

                for pic_name in pic_name_list:
                    copy_src = os.path.join(origin_data_location, pic_name)
                    copy_dst = os.path.join(each_class_save_loc, pic_name)
                    shutil.copy(copy_src, copy_dst)
                print("Collectly Copied.")

            print('-*-'*10)


def augment(target_dir=os.path.join(cwd, "experiment_0")):

    print("Augment {} datas...".format(target_dir))


    selected_mode = "rotation"
    print("Augment mode: ", selected_mode)

    train_data_location = os.path.join(target_dir, "train")


    dah = DaHandler()
    data, label = dah.imgaug_augment(train_data_location, mode=selected_mode)

    print(data.shape)
    print(label.shape)

    print(data[0])

    separete_location = os.path.basename(target_dir)
    auged_data_save_location = os.path.join(cwd, "auged_{}".format(separete_location))
    os.makedirs(auged_data_save_location, exist_ok=True)

    """
    for j, class_name in enumerate(class_list):
        print("save")
        idx = 0  # int(amount / 2)
        for i, data in enumerate(data):
            if label[i] == j:
                print(i, " | ", label[i])
                auged_class_save_location = os.path.join(auged_data_save_location, class_name)
                os.makedirs(auged_class_save_location, exist_ok=True)
                save_picture_path = os.path.join(auged_class_save_location, "{}.{}.{}.jpg".format(class_name, selected_mode, idx))

                pil_auged_img = Image.fromarray(data.astype('uint8'))  # float の場合は [0,1]/uintの場合は[0,255]で保存
                pil_auged_img.save(save_picture_path)
                idx += 1
    """


def concat():
    pass





def doWhole():

    separete()

    cwd_list = os.listdir(cwd)

    found = []
    for elem in cwd_list:
        if "experiment_" in elem:
            found.append(elem)

    print(found)

    for exp_data_dir in found:
        augment(exp_data_dir)

    concat()


def clean():

    cwd_list = os.listdir(cwd)

    found = []
    for elem in cwd_list:
        if "experiment_" in elem:
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

    parser = argparse.ArgumentParser("Data Augmentation を 100例で試そう (データ用意プログラム編)")

    parser.add_argument("--separete", action="store_true", help="ディレクトリとデータを作成します。")
    parser.add_argument("--augment", action="store_true", help="分割下データに Data Augmentation を施します。")
    parser.add_argument("--make", action="store_true", help="ディレクトリとデータを作成し水増しを行い、これら 2つのデータを結合します。")
    parser.add_argument("--clean", action="store_true", help="作成したディレクトリを削除します。")

    arg = parser.parse_args()

    if arg.separete:
        separete()
    if arg.augment:
        augment()
    elif arg.make:
        doWhole()
    elif arg.clean:
        clean()