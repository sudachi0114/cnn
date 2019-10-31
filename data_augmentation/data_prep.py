
import sys, os
sys.path.append(os.pardir)

from utils.data_separator import DataSeparator
from utils.da_handler import DaHandler

# ランダムにたくさん smaller サンプルを取るのはあとで行う。
#ds = DataSeparator()
#print(ds.__dict__)

# 様々な変換を施したデータを生成
dah = DaHandler()

picked_aug_list = ["native", "rotation", "hflip", "gnoise", "invert"]

for picked_aug in picked_aug_list:
    print("\nprocess ", picked_aug)
    dah.save_imgauged_img(save_dir=os.getcwd(), save_mode='image', aug=picked_aug)
    print("Done.")



