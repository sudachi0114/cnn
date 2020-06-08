
import os
from concatAug import concat, check

cls_list = ["cat", "dog"]


if __name__ == "__main__":

    # define
    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)
    datasets_dir = os.path.join(prj_root, "datasets")

    # data_name = os.path.join(datasets_dir, "mulSample", "m_sample")
    data_name = os.path.join(datasets_dir, "mulSample", "1000_721")

    N = 5
    for i in range(N):
        tdir = os.path.join(data_name, "sample_{}".format(i))


        mode = "train"  # "train" or "test"
    
        concat(tdir, MODE=mode)
        check(tdir+"/{}_with_aug".format(mode))
