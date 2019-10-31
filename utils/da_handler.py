
import os
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator

class DaHandler:

    def __init__(self, input_size=224, channel=3):

        self.dirs = {}
        self.dirs['cwd'] = os.getcwd()
        self.dirs['cnn_dir'] = os.path.dirname(self.dirs['cwd'])
        self.dirs['data_dir'] = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_smaller")

        self.dirs['train_dir'] = os.path.join(self.dirs['data_dir'], "train")
        self.dirs['validation_dir'] = os.path.join(self.dirs['data_dir'], "validation")
        self.dirs['test_dir'] = os.path.join(self.dirs['data_dir'], "test")

        self.train_file = os.path.join(self.dirs['data_dir'], "train.npz")
        self.validation_file = os.path.join(self.dirs['data_dir'], "validation.npz")
        self.test_file = os.path.join(self.dirs['data_dir'], "test.npz")

        # list of keras DA modes -----
        self.keras_mode_list = ['native',
                                'rotation',
                                'hflip',
                                'width_shift',
                                'height_shift',
                                'zoom',
                                'swize_center',
                                'swize_std_normalize',
                                'vflip',
                                'standard'
        ]

        # list of imgaug DA modes -----
        self.imgaug_mode_list = ['',
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
                                 #'plural'
        ]

        # attributes -----
        self.BATCH_SIZE = 10
        self.INPUT_SIZE = input_size
        self.CHANNEL = channel
        self.DO_SHUFFLE = True
        self.CLASS_MODE = 'binary'
        self.CLASS_LIST = ['cat', 'dog']


    def npzLoader(self, target_location):
        npz = np.load(target_location)
        data, label = npz['data'], npz['label']
        return data, label


    def ImageDataGeneratorForker(self, mode='native'):

        if mode == 'native':
            data_gen = ImageDataGenerator(rescale=1.0/255.0)
        elif mode == 'rotation':  # 個人的には rotation は DA の中でも効果を発揮してくれると思っている..
            data_gen = ImageDataGenerator(rescale=1.0/255.0,
                                          rotation_range=90)  # 回転 (max 90度まで)
        elif mode == 'hflip':
            data_gen = ImageDataGenerator(rescale=1.0/255.0,
                                          horizontal_flip=True)  # 左右反転
        elif mode == 'width_shift':
            data_gen = ImageDataGenerator(rescale=1.0/255.0,
                                          width_shift_range=0.125)  # 1/8 平行移動(左右)
        elif mode == 'height_shift':
            data_gen = ImageDataGenerator(rescale=1.0/255.0,
                                          height_shift_range=0.125)  # 1/8 平行移動(上下)
        elif mode == 'zoom':
            data_gen = ImageDataGenerator(rescale=1.0/255.0,
                                          zoom_range=0.2)  # (0.8 ~ 1.2 の間で) 拡大/縮小
        #elif mode == 'fwize_center':
        #    data_gen = ImageDataGenerator(rescale=1.0/255.0,
        #                                  featurewise_center=True)  # 平均を0に正規化(入力wiseに)
        elif mode == 'swize_center':
            data_gen = ImageDataGenerator(rescale=1.0/255.0,
                                          samplewise_center=True)  # 平均を0に正規化(画像1枚wiseに)
        #elif mode == 'fwize_std_normalize':
        #    data_gen = ImageDataGenerator(rescale=1.0/255.0,
        #                                  featurewise_std_normalization=True)  # 標準偏差正規化(入力wiseに)
        elif mode == 'swize_std_normalize':
            data_gen = ImageDataGenerator(rescale=1.0/255.0,
                                          samplewise_std_normalization=True)  # 標準偏差正規化(画像1枚wiseに)
        elif mode == 'vflip':
            data_gen = ImageDataGenerator(rescale=1.0/255.0,
                                          vertical_flip=True)  # 上下反転
        elif mode == 'standard':
            data_gen = ImageDataGenerator(rescale=1.0/255.0,
                                           horizontal_flip=True,
                                           width_shift_range=0.125,
                                           height_shift_range=0.125)
        else:
            print("\nError: ImageDataGeneratorForker の mode は以下のいずれかから選択してください。")
            print(self.keras_mode_list, "\n")
            raise ValueError("予期されないモードが選択されています。")

        return data_gen


    def dataGenerator(self, target_data='', mode='native'):

        train_data, train_label = self.npzLoader(self.train_file)

        data_gen = self.ImageDataGeneratorForker(mode=mode)

        if target_data == '':
            data_generator = data_gen.flow(train_data,
                                           train_label,
                                           batch_size=self.BATCH_SIZE,
                                           shuffle=self.DO_SHUFFLE)

        return data_generator


    def dataGeneratorFromDir(self, target_dir='', mode='native'):

        data_gen = self.ImageDataGeneratorForker(mode=mode)

        data_generator = data_gen.flow_from_directory(target_dir,
                                                      target_size=(self.INPUT_SIZE, self.INPUT_SIZE),
                                                      batch_size=self.BATCH_SIZE,
                                                      shuffle=self.DO_SHUFFLE,
                                                      class_mode=self.CLASS_MODE)

        return data_generator


    def getStackedData(self, target_dir='', mode='native'):

        data_generator = self.dataGeneratorFromDir(target_dir=target_dir,
                                                   mode=mode)

        iter_n = data_generator.n//self.BATCH_SIZE

        for i in range(iter_n):
            tmp_data, tmp_label = next(data_generator)
            if i == 0:
                data = tmp_data
                label = tmp_label
            else:
                data = np.vstack((data, tmp_data))
                label = np.hstack((label, tmp_label))

        return data, label


    def display_keras(self):

        for n_confirm in range(3):  # 三回出力して確認
            print("{}回目の出力".format(n_confirm+1))
            self.DO_SHUFFLE = False
            #data_generator = self.dataGenerator(mode='rotation')
            data_generator = self.dataGeneratorFromDir(target_dir=self.dirs["train_dir"], mode='rotation')

            data_checker, label_checker = next(data_generator)

            #print(data_checker[0])

            plt.figure(figsize=(12, 6))

            for i in range(len(label_checker)):
                plt.subplot(2, 5, i+1)
                plt.imshow(data_checker[i])
                plt.title("l: [{}]".format(label_checker[i]))
                plt.axis(False)

            plt.show()


    def randomDataAugument(self, num_trans):
        # 以下で定義する変換処理の内ランダムに幾つかの処理を選択
        seq = iaa.SomeOf(num_trans, [
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

    
    def imgaug_augment(self, target_dir='default',mode=''):

        if target_dir == 'default':
            data, label = self.npzLoader(self.train_file)
        else:
            data, labele = self.getStackedData(target_dir=target_dir)


        if mode == '':
            return data, label
        elif mode == 'rotation':
            imgaug_aug = iaa.Affine(rotate=(-90, 90), order=1, mode="edge")  # 90度回転
            # keras と仕様が異なることに注意
            #   keras は変化量 / imgaug は 変化の最大角を指定している
            #   開いた部分の穴埋めができない..?? mode="edge" にするとそれなり..
        elif mode == 'hflip':
            imgaug_aug = iaa.Fliplr(1.0)  # 左右反転
        elif mode == 'width_shift':
            imgaug_aug = iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge")  # 1/8 平行移動(左右)
        elif mode == 'height_shift':
            imgaug_aug = iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge")  # 1/8 平行移動(上下)
            # imgaug_aug = iaa.Crop(px=(0, 40))  <= 平行移動ではなく、切り抜き
        elif mode == 'zoom':
            imgaug_aug = iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge")  # 80~120% ズーム
            # これも keras と仕様が違って、縦横独立に拡大・縮小されるようである。
        elif mode == 'logcon':
            imgaug_aug = iaa.LogContrast(gain=(5, 15))
        elif mode == 'linecon':
            imgaug_aug = iaa.LinearContrast((0.5, 2.0))  # 明度変換
        elif mode == 'gnoise':
            imgaug_aug = iaa.AdditiveGaussianNoise(scale=[0.05*255, 0.2*255])  # Gaussian Noise
        elif mode == 'lnoise':
            imgaug_aug = iaa.AdditiveLaplaceNoise(scale=[0.05*255, 0.2*255])  # LaplaceNoise
        elif mode == 'pnoise':
            imgaug_aug = iaa.AdditivePoissonNoise(lam=(16.0, 48.0), per_channel=True)  # PoissonNoise
        elif mode == 'flatten':
            imgaug_aug = iaa.GaussianBlur(sigma=(0.5, 1.0))  # blur: ぼかし (平滑化)
        elif mode == 'sharpen':
            imgaug_aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)) # sharpen images (鮮鋭化)
        elif mode == 'emboss':
            imgaug_aug = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))  # Edge 強調
        elif mode == 'invert':
            #imgaug_aug = iaa.Invert(1.0)  # 色反転 <= これがうまく行かないので自分で作った。
            aug_data = []
            for b in range(data.shape[0]):
                aug_data.append(255-data[b])
            return np.array(aug_data), label
        elif mode == 'someof':  # 上記のうちのどれか1つ
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
        #elif mode == 'plural':  # 異なる系統の変換を複数(1つの変換あとに画素値がマイナスになるとError..)
        #    imgaug_aug = self.randomDataAugument(2)
        else:
            print("現在 imgaug で選択できる DA のモードは以下の通りです。")
            print(self.imgaug_mode_list, "\n")
            raise ValueError("予期されないモードが選択されています。")

        aug_data = imgaug_aug.augment_images(data)
        aug_data = np.clip(aug_data, 0, 255)

        # 注意: 戻り値の範囲は [0, 255] です。
        return aug_data, label



    def save_imgauged_img(self, targrt_dir='default', save_dir="prj_root", save_mode='image', aug='rotation'):

        auged_data, label = self.imgaug_augment(target_dir=targrt_dir, mode=aug)
        if save_dir == "prj_root":
            self.dirs["save_dir"] = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_auged_{}".format(aug))
        else:
            self.dirs["save_dir"] = save_dir
        os.makedirs(self.dirs["save_dir"], exist_ok=True)

        if save_mode == 'image':  # 画像として保存
            for j, class_name in enumerate(self.CLASS_LIST):
                idx = 0
                for i, data in enumerate(auged_data):
                    if label[i] == j:
                        save_dir_each =  os.path.join(save_dir, '{}'.format(class_name))
                        os.makedirs(save_dir_each, exist_ok=True)
                        save_file_cats = os.path.join(save_dir_each, "{}.{}.{}.jpg".format(class_name, aug, idx))

                        pil_auged_img = Image.fromarray(data.astype('uint8'))  # float の場合は [0,1]/uintの場合は[0,255]で保存
                        pil_auged_img.save(save_file_cats)
                        idx += 1


        elif save_mode == 'npz':  # npz file として保存
            save_file = os.path.join(save_dir, "auged_{}.npz".format(aug))
            np.save(save_file, data=auged_data, label=label)


    def display_imgaug(self, mode="rotation"):

        for n_confirm in range(3):  # 三回出力して確認
            print("{}回目の出力".format(n_confirm+1))
            self.DO_SHUFFLE = False
            data, label = self.imgaug_augment(mode=mode)
            data /= 255

            plt.figure(figsize=(12, 6))

            for i in range(10):
                plt.subplot(2, 5, i+1)
                plt.imshow(data[i])
                plt.title("l: [{}]".format(label[i]))
                plt.axis(False)

            plt.show()




if __name__ == '__main__':

    dh = DaHandler()
    """
    validation_data, validation_label = dh.npzLoader(dh.validation_file)

    print("validation_data's shape: ", validation_data.shape)
    print("validation_label's shape: ", validation_label.shape)

    test_data, test_label = dh.npzLoader(dh.test_file)

    print("test_data's shape: ", test_data.shape)
    print("test_label's shape: ", test_label.shape)

    train_data, train_label = dh.npzLoader(dh.train_file)

    print("train_data's shape: ", train_data.shape)
    print("train_label's shape: ", train_label.shape)
    data_generator = dh.dataGenerator()
    data_checker, label_checker = next(data_generator)

    print("data_checker's shape: ", data_checker.shape)
    print("label_checker's shape: ", label_checker.shape)

    #print(dh.dirs["train_dir"])
    data, labele = dh.getStackedData(target_dir=dh.dirs["train_dir"])
    print("data's shape: ", data.shape)
    print("label's shape: ", labele.shape)


    dh.display_keras()
    """
    dh.display_imgaug(mode="invert")


    #dh.save_imgauged_img(mode='image', aug='rotation')
