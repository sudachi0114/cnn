
import os

# proto-type ----------
def dirsieve(dir_list):

    ignore_files = ['.DS_Store', '__pycache__']

    for fname in ignore_files:
        if fname in dir_list:
            dir_list.remove(fname)
    dir_list = sorted(dir_list)

    return dir_list

# define ----------
cwd = os.getcwd()
prj_root = os.path.dirname(cwd)
data_dir = os.path.join(prj_root, "datasets")
data_src = os.path.join(data_dir, "small_721")

# purpose_list = os.listdir(data_src)
# purpose_list = dirsieve(purpose_list)
purpose_list = ['train', 'validation', 'test']

train_dir = os.path.join(data_src, purpose_list[0])
class_list = os.listdir(train_dir)
class_list = dirsieve(class_list)


def get_target_list(target_dir):

    print("class : ", class_list)
    print("target: ", target_dir)

    for cname in class_list:
        sub_target_dir = os.path.join(target_dir, cname)
        print(sub_target_dir)

        # sub_target_list = os.listdir(sub_target_dir)

        ## sub_target_list = []
        ## sub_target_list += os.listdir(sub_target_dir) 

        # sub_target_list = dirsieve(sub_target_list)
        # return sub_target_list

        sub_target_list = os.listdir(sub_target_dir)
        sub_target_list = dirsieve(sub_target_list)
        full_path_list = []
        for i in range(len(sub_target_list)):
            fpath = os.path.join(sub_target_dir, sub_target_list[i])
            full_path_list.append(fpath)

    ## sub_target_list = dirsieve(sub_target_list)
    ## return sub_target_list
    return full_path_list


def diterator(target_dir, batch_size):

    target_list = get_target_list(target_dir)
    # ここに shuffle 処理を条件付きで入れる

    for i in range(steps):
        begin = i * batch_size  # 0, 10, 20, 30 ...
        end = begin + batch_size
        # print(target_list[begin:end])

        dbatch = target_list[begin:end]

        yield dbatch


def recalliterator(target_dir,
                   batch_size,
                   epochs
                   # total_call,
                   # iterator=diterator):
                   ):

    target_list = get_target_list(target_dir)
    amount = len(target_list)
    steps = amount // batch_size

    total_call = steps*epochs

    for i in range(total_call):
        # if iteration counts over iterable num
        #   re-instance iterator and generator start again
        if (i % steps == 0):
            dit = diterator(target_dir, batch_size)

        batch = next(dit)
        # print(batch)
        # print(i, "(from recalliterator)")

        yield batch


# ここに dir (path) を渡したら
#    画像を配列としての読み込み, 返す機能を加える
if __name__ == "__main__":

    
    train_dir = os.path.join(data_src, "train")

    target_list = get_target_list(train_dir)  # amount 計算用
    amount = len(target_list)
    print(target_list)
    print(amount)

    set_epochs = 3
    batch_size = 10
    steps = amount // batch_size

    # it = diterator(train_dir, batch_size)
    reit = recalliterator(train_dir,
                          batch_size=batch_size,
                          epochs=set_epochs)  # 700//10 = 70, 70*3 = 210 total call
    for i in range(steps*set_epochs):
        over_batch = next(reit)
        print(i, "|", over_batch)
        print("  batch num: ", len(over_batch) )
