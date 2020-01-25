
import os, shutil

# define
cwd = os.getcwd()
tmp_src = os.path.join(cwd, "sample0")
class_list = ['cat', 'dog']

mode = "train"  # "test"
aug = "rotation"


def copy(src_dir, file_list, dist_dir, param=None):

    for pic_name in file_list:
        copy_src = os.path.join(src_dir, pic_name)
        if param is not None:
            fname, ext = pic_name.rsplit('.', 1)
            fname = "{}_".format(param) + fname
            pic_name = fname + "." + ext
            copy_dst = os.path.join(dist_dir, pic_name)
        else:
            copy_dst = os.path.join(dist_dir, pic_name)
        shutil.copy(copy_src, copy_dst)


def concat(data_src, aug="rotation"):

    target_dir = os.path.join(data_src, mode)
    print(target_dir)

    save_loc = os.path.join(data_src,
                            "{}_with_aug".format(mode))
    os.makedirs(save_loc, exist_ok=True)
    

    # copy natural target data into concat directory -----
    for i, cname in enumerate(class_list):
        sub_target_dir = os.path.join(target_dir, cname)
        sub_target_list = os.listdir(sub_target_dir)
        print(sub_target_dir)
        print("get {} data".format(len(sub_target_list)))

        # make save concated data directory -----
        sub_save_loc = os.path.join(save_loc, cname)
        os.makedirs(sub_save_loc, exist_ok=True)

        print("copy.....")
        copy(sub_target_dir, sub_target_list, sub_save_loc)
        print("    Done.")



        # copy augmented data into concat directory -----
        for i in range(2):
            print("process aug_{} ----------".format(i))
            if mode == "train":
                # auged_data_dir = "auged_train_{}".format(i)
                auged_data_dir = "{}_train_{}".format(aug, i)
            elif mode == "test":
                auged_data_dir = "{}_test_{}".format(aug, i)
            data_src = os.path.dirname(target_dir)
            auged_dir = os.path.join(data_src, auged_data_dir)

            sub_auged_dir = os.path.join(auged_dir, cname)
            sub_auged_list = os.listdir(sub_auged_dir)
            print(sub_auged_dir)
            print("get {} data".format(len(sub_auged_list)))

            print("copy.....")
            copy(sub_auged_dir, sub_auged_list, sub_save_loc, param=i)
            print("    Done.")



def rep_concat():

    cwd = os.getcwd()
    cwd_list = sorted( os.listdir(cwd) )

    sample_list = []
    for item in cwd_list:
        if "sample" in item:
            sample_list.append(item)

    if len(sample_list) == 0:
        print("結合対象のファイルはみつかりませんでした。")
    else:
        print("found:\n", (sample_list))

    for sample in sample_list:
        data_src = os.path.join(cwd, sample)
        concat(data_src, aug)



def check(save_loc):

    print("\ncheck function has executed ...")
    print(save_loc)

    target_dir = os.path.join(save_loc, "{}_with_aug".format(mode))
    
    for cname in class_list:
        sub_auged_dir = os.path.join(target_dir, cname)
        print(sub_auged_dir)
        print("  data amount: ", len( os.listdir(sub_auged_dir) ) )


def rep_check():

    print("\ncheck function has executed ...")

    cwd = os.getcwd()
    cwd_list = sorted( os.listdir(cwd) )

    samples_list = []
    for item in cwd_list:
        if "sample" in item:
            samples_list.append(item)
    
    for sample in samples_list:
        check(sample)


if __name__ == "__main__":
    # concat(tmp_src)
    # check(tmp_src)

    # rep_concat()
    rep_check()
