
# imports
import os

# define
cwd = os.getcwd()
origin_dir = os.path.join(cwd, "origin")

separeted_dir = os.path.join(cwd, "small_721")

train_dir = os.path.join(separeted_dir, "train")
class_list = os.listdir(train_dir)
ignore_files = ['.DS_Store']
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)

# red_train_dir = os.path.join(separeted_dir, "red_train")
# validation_dir = os.path.join(separeted_dir, "validation")
# test_dir = os.path.join(separeted_dir, "test")



def countAmount(dir_name):

    for porpuse in ["train", "validation", "test"]:
        pdir_name = os.path.join(dir_name, porpuse)
        for i in range(len(class_list)):
            sub_dir = os.path.join(pdir_name, class_list[i])
            print(sub_dir)
            print("  └─ ", len(os.listdir(sub_dir)))

# countAmount in `s`ingle class
def scountAmount(dir_name):

    print(dir_name)
    print("  └─ ", len(os.listdir(dir_name)))


# countAmount in `s`ingle class from keyword in file-name
def count_from_name(dir_name):

    print(dir_name)
    dir_list = os.listdir(dir_name)


    czero = 0
    cone = 0
    for i, fname in enumerate(dir_list):
        # print(i, "|", fname)
        if "cat" in fname:
            czero += 1
        elif "dog" in fname:
            cone += 1

    print("class zero: ", czero)
    print("class one : ", cone)
    


scountAmount(origin_dir)
count_from_name(origin_dir)

print("-+-"*10)
countAmount(separeted_dir)
