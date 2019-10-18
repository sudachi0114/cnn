
import os
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator

class DaHandler:

    def __init__(self):

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

        # attributes -----
        self.BATCH_SIZE = 10
        self.DO_SHUFFLE = True


    def validationNpzLoader(self):
        npz = np.load(self.validation_file)
        validation_data, validation_label = npz['data'], npz['label']
        return validation_data, validation_label

    def testNpzLoader(self):
        npz = np.load(self.test_file)
        test_data, test_label = npz['data'], npz['label']
        return test_data, test_label

    def trainNpzLoader(self):
        npz = np.load(self.train_file)
        train_data, train_label = npz['data'], npz['label']
        return train_data, train_label


    def ImageDataGeneratorForker(self, mode='native'):

        self.keras_mode_list = ['native',
                                'rotation',
                                'hflip',
                                'width_shift',
                                'height_shift',
                                'zoom',
                                'swize_center',
                                'swize_std_normalize',
                                'vflip',
                                'standard']
        print("現在 keras で選択できる DA のモードは以下の通りです。")
        print(self.keras_mode_list, "\n")


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
            raise ValueError("予期されないモードが選択されています。")

        train_data, train_label = self.trainNpzLoader()

        data_generator = data_gen.flow(train_data,
                                       train_label,
                                       batch_size=self.BATCH_SIZE,
                                       shuffle=self.DO_SHUFFLE)

        return data_generator


    def display_keras(self):

        for n_confirm in range(3):  # 三回出力して確認
            self.DO_SHUFFLE = False
            data_generator = self.ImageDataGeneratorForker(mode='rotation')

            data_checker, label_checker = next(data_generator)

            print(data_checker[0])

            plt.figure(figsize=(12, 6))

            for i in range(len(label_checker)):
                plt.subplot(2, 5, i+1)
                plt.imshow(data_checker[i])
                plt.title("l: [{}]".format(label_checker[i]))
                plt.axis(False)

            plt.show()




    def imgaug_augment(self, mode=''):

        self.imgaug_mode_list = ['',
                                 'rotation',
                                 'hflip',
                                 'width_shift',
                                 'height_shift',
                                 'zoom',
                                 'lcontrast',
                                 'gnoise',
                                 'lnoise',
                                 'pgnoise',
                                 'flatten',
                                 'sharpen',
                                 'invert',
                                 'emboss',  # 13
                                 'someof']
        print("現在 imgaug で選択できる DA のモードは以下の通りです。")
        print(self.imgaug_mode_list, "\n")

        data, label = self.trainNpzLoader()

        if mode == '':
            return data, label
        elif mode == 'rotation':
            imgaug_aug = iaa.Affine(rotate=(-90, 90), order=1, mode="edge")  # 90度回転
            # keras と仕様が異なることに注意
            #   keras は変化量 / imgaug は 変化の最大角を指定している
            #   開いた部分の穴埋めができない..?? mode="edge" にするとそれなり..
        elif mode == 'hflip':
            imgaug_aug = iaa.Fliplr(0.5)  # 左右反転
        elif mode == 'width_shift':
            imgaug_aug = iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge")  # 1/8 平行移動(左右)
        elif mode == 'height_shift':
            imgaug_aug = iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge")  # 1/8 平行移動(上下)
            # imgaug_aug = iaa.Crop(px=(0, 40))  <= 平行移動ではなく、切り抜き
        elif mode == 'zoom':
            imgaug_aug = iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge")  # 80~120% ズーム
            # これも keras と仕様が違って、縦横独立に拡大・縮小されるようである。
        #elif mode == 'scontrast':
        #    imgaug_aug = iaa.SigmoidContrast(gain=(5, 20), cutoff=(0.25, 0.75), per_channel=True)  # 彩度変換
        elif mode == 'lcontrast':
            imgaug_aug = iaa.LinearContrast((0.5, 2.0))  # 明度変換
        elif mode == 'gnoise':
            imgaug_aug = iaa.AdditiveGaussianNoise(scale=[0, 0.25*255])  # Gaussian Noise
        elif mode == 'lnoise':
            imgaug_aug = iaa.AdditiveLaplaceNoise(scale=[0, 0.25*255])  # LaplaceNoise
        elif mode == 'pnoise':
            imgaug_aug = iaa.AdditivePoissonNoise(lam=(0, 30), per_channel=True)  # PoissonNoise
        elif mode == 'flatten':
            imgaug_aug = iaa.GaussianBlur(sigma=(0, 3.0))  # blur: ぼかし (平滑化)
        elif mode == 'sharpen':
            imgaug_aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)) # sharpen images (鮮鋭化)
        elif mode == 'invert':
            imgaug_aug = iaa.Invert(p=0.2, per_channel=True)  # 色反転 (20% いずれかのチャンネルが(場合によっては複数)死ぬ)
        elif mode == 'emboss':
            imgaug_aug = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))  # Edge 強調
        elif mode == 'someof':  # 上記のうちのどれか1つ
            imgaug_aug = iaa.SomeOf(1, [
                iaa.Affine(rotate=(-90, 90), order=1, mode="edge"),
                iaa.Fliplr(0.5),
                iaa.Affine(translate_percent={"x": (-0.125, 0.125)}, order=1, mode="edge"),
                iaa.Affine(translate_percent={"y": (-0.125, 0.125)}, order=1, mode="edge"),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, mode="edge"),
                iaa.LinearContrast((0.5, 2.0)),
                iaa.AdditiveGaussianNoise(scale=[0, 0.25*255]),
                iaa.AdditiveLaplaceNoise(scale=[0, 0.25*255]),
                iaa.AdditivePoissonNoise(lam=(0, 30), per_channel=True),
                iaa.GaussianBlur(sigma=(0, 3.0)),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Invert(p=0.2, per_channel=True),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))  # 13
            ])
        else:
            raise ValueError("予期されないモードが選択されています。")

        aug_data = imgaug_aug.augment_images(data)
        aug_data = np.clip(aug_data, 0, 255)

        # 注意: 戻り値の範囲は [0, 255] です。
        return aug_data, label



    def save_imgauged_img(self, mode='image'):

        selected_aug_mode='rotation'
        aug_data, label = self.imgaug_augment(mode=selected_aug_mode)
        auged_data_dir = os.path.join(self.dirs['cnn_dir'], "dogs_vs_cats_auged_{}".format(selected_aug_mode))
        os.makedirs(auged_data_dir, exist_ok=True)

        if mode == 'image':  # 画像として保存
            j = 0
            for i, data in enumerate(aug_data):
                if label[i] == 0:
                    auged_data_dir_cat =  os.path.join(auged_data_dir, 'cat')
                    os.makedirs(auged_data_dir_cat, exist_ok=True)
                    save_file_cats = os.path.join(auged_data_dir_cat, "cat.{}.jpg".format(i))

                    pil_auged_img = Image.fromarray(data.astype('uint8'))  # float の場合は [0,1]/uintの場合は[0,255]で保存
                    pil_auged_img.save(save_file_cats)
                elif label[i] == 1:
                    auged_data_dir_dog =  os.path.join(auged_data_dir, 'dog')
                    os.makedirs(auged_data_dir_dog, exist_ok=True)
                    save_file_dogs = os.path.join(auged_data_dir_dog, "dog.{}.jpg".format(j))
                    j+=1

                    pil_auged_img = Image.fromarray(data.astype('uint8'))
                    pil_auged_img.save(save_file_dogs)

        elif mode == 'npz':  # npz file として保存
            save_file = os.path.join(auged_data_dir, "auged_{}.npz".format(selected_aug_mode))
            np.save(save_file, data=aug_data, label=label)


    def display_imgaug(self):

        for n_confirm in range(3):  # 三回出力して確認
            self.DO_SHUFFLE = False
            data, label = self.imgaug_augment(mode='rotation')
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
    validation_data, validation_label = dh.validationNpzLoader()

    print("validation_data's shape: ", validation_data.shape)
    print("validation_label's shape: ", validation_label.shape)

    test_data, test_label = dh.testNpzLoader()

    print("test_data's shape: ", test_data.shape)
    print("test_label's shape: ", test_label.shape)

    train_data, train_label = dh.trainNpzLoader()

    print("train_data's shape: ", train_data.shape)
    print("train_label's shape: ", train_label.shape)

    data_generator = dh.ImageDataGeneratorForker()
    data_checker, label_checker = next(data_generator)

    print("data_checker's shape: ", data_checker.shape)
    print("label_checker's shape: ", label_checker.shape)

    #dh.display_keras()
    dh.display_imgaug()

    dh.save_imgauged_img(mode='image')

