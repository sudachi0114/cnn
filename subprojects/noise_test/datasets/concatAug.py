
import os, shutil

# define
cwd = os.getcwd()

data_name = "medium_721"
data_src = os.path.join(cwd, data_name)

mode = "train"  # "test"
if mode == "train":
    target_dir = os.path.join(data_src, "train")
    save_data_name = "train_with_aug"
elif mode == "test":
    target_dir = os.path.join(data_src, "test")
    save_data_name = "test_with_aug"
    
save_loc = os.path.join(data_src, save_data_name)
os.makedirs(save_loc, exist_ok=True)


class_list = os.listdir(target_dir)
ignore_files = ['.DS_Store']
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)



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


def main():

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
                auged_data_dir = "auged_train_{}".format(i)
            elif mode == "test":
                auged_data_dir = "auged_test_{}".format(i)
            auged_dir = os.path.join(data_src, auged_data_dir)

            sub_auged_dir = os.path.join(auged_dir, cname)
            sub_auged_list = os.listdir(sub_auged_dir)
            print(sub_auged_dir)
            print("get {} data".format(len(sub_auged_list)))

            print("copy.....")
            copy(sub_auged_dir, sub_auged_list, sub_save_loc, param=i)
            print("    Done.")



def check():

    print("\ncheck function has executed ...")
    print(save_loc)
    
    for cname in class_list:
        sub_auged_dir = os.path.join(save_loc, cname)
        print(sub_auged_dir)
        print("  data amount: ", len( os.listdir(sub_auged_dir) ) )


if __name__ == "__main__":
    # main()
    check()
