
#
# check amount
#


import os

cls_list = ["cat", "dog"]
division_list = ["train", "validation", "test"]

# red_train_dir = os.path.join(separeted_dir, "red_train")
# validation_dir = os.path.join(separeted_dir, "validation")
# test_dir = os.path.join(separeted_dir, "test")

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

# countAmount in `s`ingle class
def dLength(DPATH):

    print(DPATH)
    dlist = os.listdir(DPATH)
    dlist = dlist_sieve(dlist)
    print("  └─ ", len(dlist))


def clsAmount(DPATH):
    for cname in cls_list:
        cdir = os.path.join(DPATH, cname)
        dLength(cdir)


def divisionAmount(DPATH):

    for division in division_list:
        print("---", division, "---")
        div_dir = os.path.join(DPATH, division)
        clsAmount(div_dir)


def count_with_key(DPATH, KEY):

    print("search [{}] into {}".format(KEY, DPATH))
    dlist = os.listdir(DPATH)
    dlist = dlist_sieve(dlist)

    cnt = 0
    for i, fname in enumerate(dlist):
        # print(i, "|", fname)
        if KEY in fname:
            cnt += 1
    print("find {}.".format(cnt))

def cls_with_key(DPATH, KEY):
    for cname in cls_list:
        cdir = os.path.join(DPATH, cname)
        count_with_key(cdir, KEY)


def division_with_key(DPATH, KEY):

    for division in division_list:
        print("---", division, "---")
        div_dir = os.path.join(DPATH, division)
        cls_with_key(div_dir, KEY)



if __name__ == "__main__":

    # define
    cwd = os.getcwd()
    datasets_dir = os.path.join(cwd, "datasets")
    origin_dir = os.path.join(datasets_dir, "origin")

    # separeted_dir = os.path.join(cwd, "small_721")
    separeted_dir = os.path.join(datasets_dir, "sample")

    divisionAmount(separeted_dir)

    # division_with_key(separeted_dir, "cat")
    # division_with_key(separeted_dir, "dog")
