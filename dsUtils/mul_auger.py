
import os, sys
sys.path.append(os.pardir)

from imgaug_auger import AugWithImgaug



if __name__ == '__main__':

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)
    datasets_dir = os.path.join(prj_root, "datasets")
    msample_dir = os.path.join(datasets_dir, "mulSample", "m_sample")

    N = 5
    for i in range(N):
        data_src = os.path.join(msample_dir, "sample_{}".format(i))
    
        train_dir = os.path.join(data_src, "train")
        validation_dir = os.path.join(data_src, "validation")
        test_dir = os.path.join(data_src, "test")

        auger = AugWithImgaug()


        for mode in ["train", "test"]:
            for i in range(2):  # aug したのを 2回 geneる
                if mode == "train":
                    dname = "auged_train_{}".format(i)
                    target_dir = train_dir
                    saug = 'plural'
                elif mode == "test":
                    dname = "auged_test_{}".format(i)
                    target_dir = test_dir
                    saug = 'fortest'

                save_loc = os.path.join(data_src, dname)
                print(save_loc)

                auger.save_imgauged_img(target_dir,
                                        INPUT_SIZE=224,
                                        SAVE_DIR=save_loc,
                                        AUGMENTATION=saug)


