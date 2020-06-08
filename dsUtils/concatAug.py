
import os, shutil

cls_list = ["cat", "dog"]


def dlist_sieve(DLIST):
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


def concat(TARGET_DIR, MODE):
    
    save_loc = os.path.join(TARGET_DIR, "{}_with_aug".format(MODE))
    os.makedirs(save_loc, exist_ok=False)

    target_dir = os.path.join(TARGET_DIR, MODE)

    # copy natural target data into concat directory
    for i, cname in enumerate(cls_list):
        cls_src_dir = os.path.join(target_dir, cname)
        cls_src_img_list = os.listdir(cls_src_dir)
        cls_src_img_list = dlist_sieve(cls_src_img_list)
        print(cls_src_dir)
        print(cls_src_img_list)
        print("get {} data".format(len(cls_src_img_list)))

        # make save concated data directory -----
        cls_save_loc = os.path.join(save_loc, cname)
        os.makedirs(cls_save_loc, exist_ok=True)

        print("copy.....")
        copy(cls_src_dir, cls_src_img_list, cls_save_loc)
        print("    Done.")



        # copy augmented data into concat directory
        for i in range(2):
            print("process aug_{} ----------".format(i))
            auged_data_dir = "auged_{}_{}".format(MODE, i)
            auged_dir = os.path.join(TARGET_DIR, auged_data_dir)

            cls_auged_dir = os.path.join(auged_dir, cname)
            cls_auged_img_list = os.listdir(cls_auged_dir)
            print(cls_auged_dir)
            print("get {} data".format(len(cls_auged_img_list)))

            print("copy.....")
            copy(cls_auged_dir, cls_auged_img_list, cls_save_loc, param=i)
            print("    Done.")



def check(DPATH):

    print("\ncheck function has executed ...")
    print(DPATH)
    
    for cname in cls_list:
        cls_auged_dir = os.path.join(DPATH, cname)
        print(cls_auged_dir)
        print("  data amount: ", len( os.listdir(cls_auged_dir) ) )


if __name__ == "__main__":

    # define
    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)
    datasets_dir = os.path.join(prj_root, "datasets")

    data_name = "1000_721"
    tdir = os.path.join(datasets_dir, data_name)


    mode = "train"  # "train" or "test"
    
    # concat(tdir, mode)
    check(tdir+"/{}_with_aug".format(mode))
