
# Data Argumentation の効果検証用プログラム
#   tl: transfer learning
#   datagen 担当

import os
import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from keras.preprocessing.image import ImageDataGenerator

# 共通
def validationDataGenerator(validation_dir, INPUT_SIZE=224, batch_size=10):

    print("\nstarting validationDataGenerator ...")

    TARGET_SIZE = (INPUT_SIZE, INPUT_SIZE)
    
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)    
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=TARGET_SIZE,
                                                                  batch_size=batch_size,
                                                                  class_mode='binary')
    return validation_generator

# 横方向平行移動
def trainDataGenerator(train_dir, INPUT_SIZE=224, batch_size=10):

    print("\nstarting trainDataGenerator ...")

    TARGET_SIZE = (INPUT_SIZE, INPUT_SIZE)
    
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                       width_shift_range=0.125)  # 1/8 平行移動
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=TARGET_SIZE,
                                                        batch_size=batch_size,
                                                        class_mode='binary')
    return train_generator

    
if __name__ == '__main__':

    current_location = os.path.abspath(__file__)  # このファイルの絶対パスを取得
    cwd, base_name = os.path.split(current_location)  # path と ファイル名に分割
    file_name, _ = os.path.splitext(base_name)  # ファイル名と拡張子を分離
    print("current location : ", cwd, ", this file : ", file_name)

    cnn_dir = os.path.dirname(cwd)
    
    # 少ないデータに対して水増しを行いたいので smaller を選択
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    train_dir = os.path.join(data_dir, "train")  # global
    validation_dir = os.path.join(data_dir, "validation")  # global
    print("train data is in ... ", train_dir)
    print("validation data is in ...", validation_dir)

        
    validation_generator = validationDataGenerator(validation_dir)
    train_generator = trainDataGenerator(train_dir)

 
    data_checker, label_checker = next(train_generator)
    print("data shape : ", data_checker.shape)
    print("label shape : ", label_checker.shape)

