
# imgaug library で画像を変換して遊ぼう。

import os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

class playImgaug():

    def __init__(self):

        self.dirs = {}
        self.dirs['cwd'] = os.getcwd()
        self.dirs['cnn_dir'] = os.path.dirname(self.dirs['cwd'])
        self.dirs['data_dir'] = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_smaller")
        self.target_file = os.path.join(self.dirs['data_dir'], "train.npz")


    def get_image(self, img_mode='wallaby'):

        print("Image mode is ", img_mode)
        if img_mode == 'wallaby':
            wallaby = ia.quokka()
            print("wallaby shape: ", wallaby.shape)
            return wallaby
        elif img_mode == 'mypic':
            npz = np.load(self.target_file)
            data = npz['data']
            retval = data[5]/255.0
            print("my input image's shape: ", retval.shape)
            return retval


    def doAugments(self, img_mode='wallaby', aug_mode='Dropout'):

        x = self.get_image(img_mode)

        print("Augment is ", aug_mode)
        if aug_mode == 'dropout':
            aug = iaa.Dropout(p=0.5)
        elif aug_mode == 'cdropout':
            aug = iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)
        elif aug_mode == 'invert':
            aug = iaa.Invert(0.5, per_channel=0.75)
        elif aug_mode == 'crop':
            # px=(min, max)
            #   min ~ max pixel 分だけ Crop される
            #   # 縦/横 に対して 両側から何 pixel Crop するのか
            #   # つまり Crop 幅は2倍になる
            aug = iaa.Crop(px=(1,20), keep_size=True)  # 元の大きさを保つ場合は keep_size=True
        elif aug_mode == 'hflip':
            aug = iaa.Fliplr(0.5)  # 50% の確率で画像を左右反転
        elif aug_mode == 'vflip':
            aug = iaa.Flipud(0.5)  # 50% の確率で画像を上下反転
        elif aug_mode == 'gblur':
            aug = iaa.GaussianBlur(sigma=(0, 3.0))  # blur: ぼかし
        elif aug_mode == 'ablur':
            aug = iaa.AverageBlur(k=(2, 7))
        elif aug_mode == 'mblur':
            aug = iaa.MedianBlur(k=(3, 11))
            # 2回目で Error
        elif aug_mode == 'edetect':  # 微分フィルタ
            aug = iaa.EdgeDetect(alpha=(0.5, 1.0))
        elif aug_mode == 'dedetect':
            aug = iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0))
        elif aug_mode == 'sharpen':  # 鮮鋭化??
            aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)) # sharpen images
        elif aug_mode == 'emboss':
            aug = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)) # emboss images
        elif aug_mode == 'add':
            aug = iaa.Add((-1, 1), per_channel=0.5) # change brightness of images (by -1 to 1 of original val)
        elif aug_mode == 'addhs':
            aug = iaa.AddToHueAndSaturation((-2, 2)) # change hue and saturation
            # Error
        elif aug_mode == 'mul':
            aug = iaa.Multiply((0.5, 1.5), per_channel=0.5)

        aug_x = aug.augment_image(x)
        print("augmented image shape: ", aug_x.shape)

        return aug_x


    def display(self, img_mode='wallaby', aug_mode='Dropout'):

        x = self.doAugments(img_mode, aug_mode)

        plt.imshow(x)
        #plt.imshow(x.astype(np.uint8))
        plt.show()



if __name__ == '__main__':

    playImgaug = playImgaug()
    #playImgaug.display()
    #playImgaug.display(img_mode='mypic')

    for i in range(5):
        print("display", i, "times -----")
        playImgaug.display(img_mode='mypic', aug_mode='mul')
