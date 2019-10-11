
# 誤認識した画像をプロットして考察する:
#   predict の内容を確かめる


import os, datetime
now = datetime.datetime.now()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#server_flg = False
import tensorflow as tf
#if tf.test.is_gpu_available():
server_flg = True
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam

def testDataGenerator(test_dir, input_size, batch_size=10):

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(input_size, input_size),
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      class_mode='binary')
    return test_generator



def reloadModel(child_log_dir):

    child_log_list = os.listdir(child_log_dir)

    for f in child_log_list:
        if "model" in f:
            model_file = os.path.join(child_log_dir, f)
    print("Use saved model : ", model_file)

    model = load_model(model_file, compile=True)

    return model



def predStocker(generator, model, batch_size=10):

    pred_steps = generator.n//batch_size    

    # predict -----
    pred_result = model.predict_generator(generator,
                                          steps=pred_steps,
                                          verbose=1)


    # stock -----
    pred_target, labels = [], []
    for i in range(pred_steps):
        tmp_target, tmp_label = next(generator)
        if i == 0:
            pred_target = tmp_target
            labels = tmp_label
        else:
            pred_target = np.vstack((pred_target, tmp_target))
            labels = np.hstack((labels, tmp_label))


    return pred_result, pred_target, labels  # 最初のクラスの予測値(確率)のnp配列, 予測target(画像), 正解label



def main():

    # 先にモデルを読み込み、モデルの input_shape を test_generator に渡す
    model = reloadModel(child_log_dir)
    #model.summary()    

    input_size = model.inputs[0].shape[1]
    test_generator = testDataGenerator(test_dir, input_size=input_size)


    pred_result, pred_target, labels = predStocker(test_generator, model)
    #print("pred_result : ", pred_result)
    #print(labels)

    labels_class = []
    for i in range(len(labels)):
        if labels[i] == 0:
            labels_class.append('cat')
        elif labels[i] == 1:
            labels_class.append('dog')

    #pred = pd.DataFrame(pred_result, columns=['cat'])
    #pred['dog'] = 1.0 - pred['cat']
    pred = pd.DataFrame(pred_result, columns=['dog'])
    pred['cat'] = 1.0 - pred['dog']
    pred['class'] = pred.idxmax(axis=1)
    pred['label'] = labels_class
    pred['collect'] = (pred['class'] == pred['label'])
    print(pred)

    confuse = pred[pred['collect'] == False].index.tolist()
    print("\nwrong recognized indeices are ", confuse)
    print("wrong recognized amount is ", len(confuse))
    print("wrong rate : ", 100*len(confuse)/len(labels), "%")

    # wrong rate の check 用
    print("\ncheck secence ...")
    score = model.evaluate(pred_target, labels, verbose=1)
    print("test accuracy: ", score[1])
    print("test wrong rate must be (1-accuracy): ", 1.0-score[1])
    

    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(hspace=0.5)
    #plt.title("Confusion picures #={}".format(len(confuse)))

    for i in range(len(pred_target)):
        plt.subplot(5, 10, 1+i)
        plt.imshow(pred_target[i])
        plt.axis(False)
        plt.title("pred:{}".format(pred['class'][i]))
    if server_flg:
        img_file_place = os.path.join(child_log_dir, "{0}_AllPics_{1:%y%m%d}_{2:%H%M}.png".format(selected_child_log_dir, now, now))
        plt.savefig(img_file_place)
    else:
        plt.show()


if __name__ == '__main__':

    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)

    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    test_dir = os.path.join(data_dir, "test")
    print("test dir is in ... ", test_dir)
    #print(len(os.listdir(test_dir)))  # 2 (dog/cat)

    log_dir = os.path.join(cwd, "log")
    child_log_list = os.listdir(log_dir)
    
    print("\nfind logs below -----")
    for i, child in enumerate(child_log_list):
        print(i, " | ", child)
    print("\nPlease chose one child_log by index ...")
    selected_child_log_idx = input(">>> ")
    
    selected_child_log_dir = child_log_list[int(selected_child_log_idx)]
    child_log_dir = os.path.join(log_dir, selected_child_log_dir)
    
    print("\nuse log at ", child_log_dir, "\n")
    print("this directory contain : ", os.listdir(child_log_dir))  # log list [history.pkl, model&weights.h5, log]

    
    main()
    
