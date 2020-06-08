
import os
from concatAug import concat, check

cls_list = ["cat", "dog"]


if __name__ == "__main__":

    # define
    cwd = os.getcwd()
    datasets_dir = os.path.dirname(cwd)

    data_name = os.path.join(datasets_dir, "mulSample", "m_sample")

    N = 3
    for i in range(N):
        tdir = os.path.join(data_name, "sample_{}".format(i))


        mode = "train"  # "train" or "test"
    
        concat(tdir, MODE=mode)
        check(tdir+"/{}_with_aug".format(mode))
