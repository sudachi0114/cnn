
import sys, os
sys.path.append(os.pardir)

from utils.data_separator import DataSeparator
from utils.da_handler import DaHandler

# ランダムにたくさん smaller サンプルを取るのはあとで行う。
#ds = DataSeparator()
#print(ds.__dict__)

# 様々な変換を施したデータを生成
dah = DaHandler()

cwd = os.getcwd()

picked_aug_list = ["native", "rotation", "hflip", "gnoise", "invert"]


def daImgCreate():

    for picked_aug in picked_aug_list:
        print("\nprocess ", picked_aug, " .....")
        dah.save_imgauged_img(save_dir=cwd, save_mode='image', aug=picked_aug)
        print("Done.")

def daFolderConcatenate():

    raw_cwd_list = os.listdir(cwd)
    #print("raw: ", raw_cwd_list)

    auged_datas = []
    for canditate in raw_cwd_list:
         #print("canditate: ", canditate)
        if "dogs_vs_cats" in canditate:
            auged_datas.append(canditate)

    #print("processed: ", auged_datas)
    #print("len: ", len(auged_datas))

    #print(picked_aug_list)
    #print(2**len(picked_aug_list))
    #print( format(2**len(picked_aug_list), 'b') )

    combination = {}

    for i in range(2**len(picked_aug_list)):
        kanri_no = format(i, '05b')
        print(kanri_no)
        selected = []
        for j in range(len(kanri_no)):
            if kanri_no[j] == "1":
                selected.append(picked_aug_list[j])
                print("    select: ", picked_aug_list[j])
        combination[kanri_no] = selected

    print(combination)





if __name__ == '__main__':

    # daImageCreate()
    daFolderConcatenate()




