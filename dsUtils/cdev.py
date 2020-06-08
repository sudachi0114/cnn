
#
# the program of separate images
#    into each class
#

# imports
import os, shutil

cls_list = ["cat", "dog"]
division_list = ["train", "validation", "test"]


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


def countAmount(DIR_PATH, CLS_LIST):

    for i, cname in enumerate(CLS_LIST):
        cdir = os.path.join(DIR_PATH, cname)
        print(cdir)
        print("  └─ ", len(os.listdir(cdir)))



def copy(SRC_DIR, FILE_LIST, DIST_DIR, param=None):

    print("copy from {} data".format(SRC_DIR))
    print("  -> to {} .....".format(DIST_DIR))
    print("    amount: ", len(FILE_LIST))

    for pic_name in FILE_LIST:
        copy_src = os.path.join(SRC_DIR, pic_name)
        if param is not None:
            fname, ext = pic_name.rsplit('.', 1)
            fname = "{}_".format(param) + fname
            pic_name = fname + "." + ext
            copy_dst = os.path.join(DIST_DIR, pic_name)
        else:
            copy_dst = os.path.join(DIST_DIR, pic_name)
        shutil.copy(copy_src, copy_dst)

    print("  Done.")




def execute(TARGET_DIR, CLS_LIST):

    # dir_list = os.listdir(origin_dir)
    dir_list = os.listdir(TARGET_DIR)
    dir_list = dlist_sieve(dir_list)

    par_datasets_dir = os.path.dirname(TARGET_DIR)

    # class devided ships net directory -----
    cdev_origin = os.path.join(par_datasets_dir,
                               "cdev_origin")
    os.makedirs(cdev_origin, exist_ok=False)

    for i, cname in enumerate(cls_list):
        cdir = os.path.join(cdev_origin, cname)
        os.makedirs(cdir, exist_ok=True)

        imgs_list = []
        for fname in dir_list:
            if cname in fname:
                imgs_list.append(fname)

        copy(origin_dir, imgs_list, cdir)

    # check:
    countAmount(cdev_origin, cls_list)



if __name__ == "__main__":

    # define
    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)
    datasets_dir = os.path.join(prj_root, "datasets")

    origin_dir = os.path.join(datasets_dir, "origin")

    print(origin_dir)
    print("origin image amount:", len(os.listdir(origin_dir)))

    execute(origin_dir, cls_list)
    countAmount(datasets_dir+"/cdev_origin", cls_list)

