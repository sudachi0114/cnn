
import os, sys
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import imgaug as ia
import imgaug.augmenters as iaa

from utils.img_utils import inputDataCreator


class AugWithImgaug:

    def __init__(self, INPUT_SIZE=224, CHANNEL=3, BATCH_SIZE=10):

        # 最低限の dir 構成を保持
        self.dirs = {}
        self.dirs['cwd'] = os.getcwd()
        self.dirs['prj_root'] = os.path.dirname(self.dirs['cwd'])
        self.dirs['datasets_dir'] = os.path.join(self.dirs['prj_root'], "datasets")
    

        # list of imgaug DA modes -----
        self.imgaug_aug_list = ['native',
                                'rotation',
                                'hflip',
                                'width_shift',
                                'height_shift',
                                'zoom',
                                'logcon',
                                'linecon',
                                'gnoise',
                                'lnoise',
                                'pnoise',
                                'flatten',
                                'sharpen',
                                'invert',
                                'emboss',  # 14
                                'someof',
                                'plural',
                                'fortest'
        ]

        # attributes -----
        self.batch_size = BATCH_SIZE
        self.input_size= INPUT_SIZE
        self.channel = CHANNEL
        self.DO_SHUFFLE = True # ?? FIXME (WHO AM I ?)
        self.cls_list = ['cat', 'dog']


    def img2array(self, TARGET_DIR, INPUT_SIZE, NORMALIZE=False):

        data, label = inputDataCreator(TARGET_DIR,
                                       INPUT_SIZE,
                                       normalize=NORMALIZE)
        return data, label



    def randomDataAugument(self, NUM_TRANS):
        # 以下で定義する変換処理の内ランダムに幾つかの処理を選択
        seq = iaa.SomeOf(NUM_TRANS, [
            iaa.Affine(rotate=(-90, 90), order=1, mode="edge"),
            iaa.Fliplr(1.0),
            iaa.OneOf([
                # 同じ系統の変換はどれか1つが起きるように 1つにまとめる
                iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge"),
                iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge")
            ]),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge"),
            iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=[0.05 * 255, 0.2 * 255]),
                iaa.AdditiveLaplaceNoise(scale=[0.05 * 255, 0.2 * 255]),
                iaa.AdditivePoissonNoise(lam=(16.0, 48.0), per_channel=True)
            ]),
            iaa.OneOf([
                iaa.LogContrast((0.5, 1.5)),
                iaa.LinearContrast((0.5, 2.0))
            ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.5, 1.0)),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
            ]),
            iaa.Invert(1.0)
        ], random_order=True)

        return seq


    def randomTestAugument(self, NUM_TRANS):
        # 本番環境の危機から取得したデータにノイズが乗っていた状態を想定して変換
        seq = iaa.SomeOf(NUM_TRANS, [
            iaa.Affine(rotate=(-90, 90), order=1, mode="edge"),
            iaa.Fliplr(1.0),
            iaa.OneOf([
                # 同じ系統の変換はどれか1つが起きるように 1つにまとめる
                iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge"),
                iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge")
            ]),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge"),
            iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=[0.05 * 255, 0.2 * 255]),
                iaa.AdditiveLaplaceNoise(scale=[0.05 * 255, 0.2 * 255]),
                iaa.AdditivePoissonNoise(lam=(16.0, 48.0), per_channel=True)
            ]),
            iaa.OneOf([
                iaa.LogContrast((0.5, 1.5)),
                iaa.LinearContrast((0.5, 2.0))
            ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.5, 1.0)),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
            ]),
            # iaa.Invert(1.0)  # 色反転は故障しすぎでしょう..
        ], random_order=True)

        return seq


    def imgaug_augment(self, TARGET_DIR, INPUT_SIZE, NORMALIZE=False, AUGMENTATION='native'):

        data, label = self.img2array(TARGET_DIR, INPUT_SIZE, NORMALIZE)

        if AUGMENTATION == 'native':
            return data, label
        elif AUGMENTATION == 'rotation':
            imgaug_aug = iaa.Affine(rotate=(-90, 90), order=1, mode="edge")  # 90度 "まで" 回転
        elif AUGMENTATION == 'hflip':
            imgaug_aug = iaa.Fliplr(1.0)  # 左右反転
        elif AUGMENTATION == 'width_shift':
            imgaug_aug = iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge")  # 1/8 平行移動(左右)
        elif AUGMENTATION == 'height_shift':
            imgaug_aug = iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge")  # 1/8 平行移動(上下)
            # imgaug_aug = iaa.Crop(px=(0, 40))  <= 平行移動ではなく、切り抜き
        elif AUGMENTATION == 'zoom':
            imgaug_aug = iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge")  # 80~120% ズーム
            # これも keras と仕様が違って、縦横独立に拡大・縮小されるようである。
        elif AUGMENTATION == 'logcon':
            imgaug_aug = iaa.LogContrast((0.5, 1.5))
        elif AUGMENTATION == 'linecon':
            imgaug_aug = iaa.LinearContrast((0.5, 2.0))  # 明度変換
        elif AUGMENTATION == 'gnoise':
            imgaug_aug = iaa.AdditiveGaussianNoise(scale=[0.05*255, 0.2*255])  # Gaussian Noise
        elif AUGMENTATION == 'lnoise':
            imgaug_aug = iaa.AdditiveLaplaceNoise(scale=[0.05*255, 0.2*255])  # LaplaceNoise
        elif AUGMENTATION == 'pnoise':
            imgaug_aug = iaa.AdditivePoissonNoise(lam=(16.0, 48.0), per_channel=True)  # PoissonNoise
        elif AUGMENTATION == 'flatten':
            imgaug_aug = iaa.GaussianBlur(sigma=(0.5, 1.0))  # blur: ぼかし (平滑化)
        elif AUGMENTATION == 'sharpen':
            imgaug_aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)) # sharpen images (鮮鋭化)
        elif AUGMENTATION == 'emboss':
            imgaug_aug = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))  # Edge 強調
        elif AUGMENTATION == 'invert':
            imgaug_aug = iaa.Invert(1.0)  # 色反転 <= これがうまく行かないので自分で作った。
        elif AUGMENTATION == 'someof':  # 上記のうちのどれか1つ
            imgaug_aug = iaa.SomeOf(1, [
                iaa.Affine(rotate=(-90, 90), order=1, mode="edge"),
                iaa.Fliplr(1.0),
                iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge"),
                iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge"),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge"),
                iaa.LogContrast((0.5, 1.5)),
                iaa.LinearContrast((0.5, 2.0)),
                iaa.AdditiveGaussianNoise(scale=[0.05*255, 0.25*255]),
                iaa.AdditiveLaplaceNoise(scale=[0.05*255, 0.25*255]),
                iaa.AdditivePoissonNoise(lam=(16.0, 48.0), per_channel=True),
                iaa.GaussianBlur(sigma=(0.5, 1.0)),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                iaa.Invert(1.0)  # 14
            ])
        elif AUGMENTATION == 'plural':  # 異なる系統の変換を複数(1つの変換あとに画素値がマイナスになるとError..)
            imgaug_aug = self.randomDataAugument(2)
        elif AUGMENTATION == 'fortest':  # plural - invert (色反転) (test 用)
            imgaug_aug = self.randomTestAugument(2)
        else:
            print("現在 imgaug で選択できる DA のモードは以下の通りです。")
            print(self.imgaug_aug_list, "\n")
            raise ValueError("予期されないモードが選択されています。")

        aug_data = imgaug_aug.augment_images(data)
        aug_data = np.clip(aug_data, 0, 255)

        return aug_data, label



    def save_imgauged_img(self, TARGET_DIR, INPUT_SIZE, NORMALIZE=False,
                          SAVE_DIR=None, AUGMENTATION='rotation'):

        auged_data, label = self.imgaug_augment(TARGET_DIR=TARGET_DIR,
                                                INPUT_SIZE=INPUT_SIZE,
                                                NORMALIZE=NORMALIZE,
                                                AUGMENTATION=AUGMENTATION)
        if SAVE_DIR is None:
            # origin_dir = os.path.basename(TARGET_DIR)
            # /home/user/cnn/datasets/some_721/train => [cat, dog]
            #   par_dir   : /home/user/cnn/datasets/some_721/
            #   dataset_name: auged_train
            par_dir, dataset_name = os.path.split(TARGET_DIR)
            save_dir = os.path.join(par_dir,
                                    "{}_{}".format(AUGMENTATION, dataset_name))
        else:
            save_dir = SAVE_DIR
        os.makedirs(save_dir, exist_ok=False)

        for j, cname in enumerate(self.cls_list):
            idx = 0
            for i, data in enumerate(auged_data):
                if label[i] == j:
                    cls_save_dir =  os.path.join(save_dir, cname)
                    os.makedirs(cls_save_dir, exist_ok=True)
                    save_file = os.path.join(cls_save_dir, "{}.{}.{}.jpg".format(cname, AUGMENTATION, idx))

                    pil_auged_img = Image.fromarray(data.astype('uint8'))  # float の場合は [0,1]/uintの場合は[0,255]で保存
                    pil_auged_img.save(save_file)
                    idx += 1


    def display_imgaug(self, TARGET_DIR, INPUT_SIZE, NORMALIZE=False, AUGMENTATION="rotation"):

        for n_confirm in range(3):  # 三回出力して確認
            print("{}回目の出力".format(n_confirm+1))
            self.DO_SHUFFLE = False
            data, label = self.imgaug_augment(TARGET_DIR,
                                              INPUT_SIZE,
                                              NORMALIZE,
                                              AUGMENTATION=AUGMENTATION)
            data = data / 255

            plt.figure(figsize=(12, 6))

            for i in range(10):
                plt.subplot(2, 5, i+1)
                plt.imshow(data[i])
                plt.title("l: [{}]".format(label[i]))
                plt.axis(False)

            plt.show()




if __name__ == '__main__':

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)
    datasets_dir = os.path.join(prj_root, "datasets")
    data_src = os.path.join(datasets_dir, "1000_721")
    
    train_dir = os.path.join(data_src, "train")
    validation_dir = os.path.join(data_src, "validation")
    test_dir = os.path.join(data_src, "test")

    auger = AugWithImgaug()

    """
    train_data, train_label = auger.img2array(train_dir, 224, NORMALIZE=False)
    print(train_data.shape)
    print(train_label.shape)

    auged_data, label = auger.imgaug_augment(train_dir, 224, NORMALIZE=False, AUGMENTATION="invert")
    print(auged_data.shape)
    print(label.shape)

    auger.display_imgaug(train_dir, 224, NORMALIZE=False, AUGMENTATION="plural")
    """


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


