
# Data Argumentation の効果検証用プログラム
#   train 担当

import os, pickle
import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from my_model import build_model
#from datagen import trainDataGenerator, validationDataGenerator
#from native_learn import trainDataGenerator, validationDataGenerator
#from width_shift import trainDataGenerator, validationDataGenerator
#from height_shift import trainDataGenerator, validationDataGenerator
#from st_arg import trainDataGenerator, validationDataGenerator
from horizontal_flip import trainDataGenerator, validationDataGenerator
da_name = 'horizontal_flip'


def train(set_epochs=50):

    validation_generator = validationDataGenerator(validation_dir)
    train_generator = trainDataGenerator(train_dir)
 
    data_checker, label_checker = next(train_generator)
    print("data shape : ", data_checker.shape)
    print("label shape : ", label_checker.shape)

    #print(data_checker[1])
    #print(label_checker[1])

    INPUT_SIZE = data_checker.shape[1]
    print("INPUT_SIZE: ", INPUT_SIZE)
    
    CHANNEL = data_checker.shape[3]
    print("set channel : ", CHANNEL)

    batch_size = data_checker.shape[0]
    print("batch_size : ", batch_size)

    model = build_model(INPUT_SIZE, CHANNEL)

    model.summary()
    

    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")


    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=set_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  verbose=1)

    # save model & weights
    model_file = os.path.join(child_log_dir, '{}_model.h5'.format(da_name))
    model.save(model_file)

    # save history
    history_file = os.path.join(child_log_dir, '{}_history.pkl'.format(da_name))
    with open(history_file, 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)

    
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

    # make log dir -----
    log_dir = os.path.join(cwd, 'log')
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, da_name)
    os.makedirs(child_log_dir, exist_ok=True)

    
    train()
