
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
data_src = os.path.join(cwd, "small_721")

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

        sub_target_list = os.listdir(sub_target_dir)
        # target_list = dirsieve(target_list)

        amount = len(sub_target_list)

        return sub_target_list


def diterator(target_list, batch_size):

    for i in range(steps):
        begin = i * batch_size # 0, 10, 20, 30 ...
        end = begin + batch_size
        # print(target_list[begin:end])

        yield target_list[begin:end]
    

if __name__ == "__main__":

    target_list = get_target_list(train_dir)

    batch_size = 10
    steps = 700 // batch_size

    for i in range(steps):
        batch = next(diterator(target_list, batch_size))
        print(batch)
