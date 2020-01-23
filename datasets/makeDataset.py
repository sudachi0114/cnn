
# imports
import os, shutil

# define
cwd = os.getcwd()

"""
train_dir = os.path.join(cwd, "train")
class_list = os.listdir(train_dir)
ignore_files = ['.DS_Store']
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)

red_train_dir = os.path.join(cwd, "red_train")
validation_dir = os.path.join(cwd, "validation")
test_dir = os.path.join(cwd, "test")
"""

class_list = ["cat", "dog"]

origin_dir = os.path.join(cwd, "origin")

# class devided ships net directory -----
cdev_origin = os.path.join(cwd, "cdev_origin")
os.makedirs(cdev_origin, exist_ok=True)


def countAmount(dir_name):

    for i in range(len(class_list)):
        sub_dir = os.path.join(dir_name, class_list[i])
        print(sub_dir)
        print("  └─ ", len(os.listdir(sub_dir)))


def copy(src_dir, file_list, dist_dir, param=None):

    print( "copy from {} data".format(src_dir) )
    print( "  -> to {} .....".format(dist_dir) )
    print( "    amount: ", len(file_list) )

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

    print( "  Done." )




def main():

    dir_list = os.listdir(origin_dir)

    for i, cname in enumerate(class_list):
        sub_dir = os.path.join(cdev_origin, cname)
        os.makedirs(sub_dir, exist_ok=True)

        sub_list = []
        for fname in dir_list:
            if cname in fname:
                sub_list.append(fname)

        copy(origin_dir, sub_list, sub_dir)


def check():

    for cname in class_list:
        target_dir = os.path.join(cdev_origin, cname)

        print(target_dir)
        print("  data amount: ", len( os.listdir(target_dir) ) )


if __name__ == "__main__":
    # main()
    check()
