import os, sys, shutil
sys.path.append(os.pardir)

from PIL import Image
#from utils.da_handler import DaHandler
from utils.aug_with_imgaug import AugWithImgaug

cwd = os.getcwd()
prj_root = os.path.dirname(cwd)

origin_data_location = os.path.join(prj_root, "dogs_vs_cats_origin")

class_list = ['cat', 'dog']

amount = len(os.listdir(origin_data_location))


def separete():

    data_size = 2000
    # define
    # train_size(rate) = 0.7
    # validation_size = 0.2
    # test_size = 0.1


    sample_num = int( amount / data_size )
    # 25000 / 1000 => 25 sample

    print("total: ", amount)
    print("data size: ", data_size)
    print("sample num: ", sample_num)


    for i in range(sample_num):
        save_location = os.path.join(cwd, "experiment_{}".format(i))
        os.makedirs(save_location, exist_ok=True)

        each_class_data_size = int( data_size / 2 )
        print("  each class's data size: ", each_class_data_size)

        idx_start = i * each_class_data_size
        idx_end = idx_start + each_class_data_size

        
        for class_name in class_list:

            each_class_save_loc = os.path.join(save_location, class_name)
            os.makedirs(each_class_save_loc, exist_ok=True)
            print("\nmake directory: ", each_class_save_loc)

            pic_name_list = []

            print("  Amount of {} pictures is : {}".format(class_name, each_class_data_size))
            print("  data range is from {} to {}".format(idx_start, idx_end))

            for i in range(idx_start, idx_end):
                pic_name_list.append("{}.{}.jpg".format(class_name, i))

            print("    !! check squence: ")
            assert len(pic_name_list) == each_class_data_size
            print("    !!  -> cleared.")

            for pic_name in pic_name_list:
                copy_src = os.path.join(origin_data_location, pic_name)
                copy_dst = os.path.join(each_class_save_loc, pic_name)
                shutil.copy(copy_src, copy_dst)
            print("Collectly Copied.")

        print('-*-'*10)


def augment(target_dir):

    print("Augment {} datas...".format(target_dir))


    selected_mode = "rotation"
    print("Augment mode: ", selected_mode)

    #train_data_location = os.path.join(target_dir, "train")

    auger = AugWithImgaug()
    data, label = auger.imgaug_augment(target_dir,
                                       224,
                                       normalize=False,
                                       aug=selected_mode)

    print("data shape: ", data.shape)
    print("label shape: ", label.shape)

    save_data_shape = data[0].shape

    #data *= 255

    separete_location = os.path.basename(target_dir)
    auged_data_save_location = os.path.join(cwd, "auged_{}".format(separete_location))
    os.makedirs(auged_data_save_location, exist_ok=True)

    for j, class_name in enumerate(class_list):
        print("\nsave {} class after generation".format(class_name))
        idx = 0  # int(amount / 2)
        for i, each_data in enumerate(data):
            if label[i] == j:
                auged_class_save_location = os.path.join(auged_data_save_location, class_name)
                os.makedirs(auged_class_save_location, exist_ok=True)
                save_picture_path = os.path.join(auged_class_save_location, "{}.{}.{}.jpg".format(class_name, selected_mode, idx))

                assert each_data.shape == save_data_shape
                pil_auged_img = Image.fromarray(each_data.astype('uint8'))  # float の場合は [0,1]/uintの場合は[0,255]で保存
                pil_auged_img.save(save_picture_path)
                idx += 1
        print("Done.")
    print("Collectly Saved.")



def concat(normal_data_dir, auged_data_dir):

    separete_location = os.path.basename(normal_data_dir)
    concat_data_save_location = os.path.join(cwd, "concat_{}".format(separete_location))
    os.makedirs(concat_data_save_location, exist_ok=True)

    #train_location = os.path.join(normal_data_dir, "train")

    for class_name in class_list:
        each_class_save_location = os.path.join(concat_data_save_location, class_name)
        os.makedirs(each_class_save_location, exist_ok=True)
        print("\nmake directory: ", each_class_save_location)

        each_class_normal_data = os.path.join(normal_data_dir, class_name)
        each_class_auged_data = os.path.join(auged_data_dir, class_name)


        copy_list = []

        for moto_img in os.listdir(each_class_normal_data):
            copy_list.append( os.path.join(each_class_normal_data, moto_img) )
        for auged_img in os.listdir(each_class_auged_data):
            copy_list.append( os.path.join(each_class_auged_data, auged_img) )

        print(copy_list)

        for pic_location in copy_list:
            copy_src = pic_location
            copy_dst = os.path.join(each_class_save_location)
            shutil.copy(copy_src, copy_dst)
        print("Collectly Concated.")

        print("\n----------\n")




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

    """ このデータの持ち方だと concat があまり意味を為さない
    auged_cwd_list = os.listdir(cwd)

    auged_found = []
    for elem in auged_cwd_list:
        if "auged_" in elem:
            auged_found.append(elem)
    print(auged_found)


    for i, base_dir in enumerate(found):
        auged_dir = "auged_{}".format(os.path.basename(base_dir))
        concat(base_dir, auged_dir)
    """


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

    parser = argparse.ArgumentParser(description="Data Augmentation を 100例で試そう (データ用意プログラム編)")

    parser.add_argument("--separete", action="store_true", help="ディレクトリとデータを作成します。")
    parser.add_argument("--augment", action="store_true", help="分割下データに Data Augmentation を施します。")
    parser.add_argument("--concat", action="store_true", help="DA したデータと。")
    parser.add_argument("--make", action="store_true", help="ディレクトリとデータを作成し水増しを行い、これら 2つのデータを結合します。")
    parser.add_argument("--clean", action="store_true", help="作成したディレクトリを削除します。")

    arg = parser.parse_args()

    if arg.separete:
        separete()
    if arg.augment:
        augment()  # os.path.join(cwd, "experiment_0")
    elif arg.concat:
        concat()  # os.path.join(cwd, "experiment_0") / os.path.join(cwd, "auged_experiment_0")
    elif arg.make:
        doWhole()
    elif arg.clean:
        clean()
