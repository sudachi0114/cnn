
# 誤認識した画像をプロットして考察する:


import os, datetime
now = datetime.datetime.now()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


def testDataGenerator(test_dir, input_size=150, batch_size=10):

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

    model = load_model(model_file, compile=False)

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

    test_generator = testDataGenerator(test_dir)

    model = reloadModel(child_log_dir)
    #model.summary()

    pred_result, pred_target, labels = predStocker(test_generator, model)
    #print("pred_result : ", pred_result)
    #print(labels)

    labels_class = []
    for i in range(len(labels)):
        if labels[i] == 0:
            labels_class.append('cat')
        elif labels[i] == 1:
            labels_class.append('dog')

    pred = pd.DataFrame(pred_result, columns=['cat'])
    pred['dog'] = 1.0 - pred['cat']
    pred['class'] = pred.idxmax(axis=1)
    pred['label'] = labels_class
    pred['collect'] = (pred['class'] == pred['label'])
    print(pred)

    confuse = pred[pred['collect'] == False].index.tolist()
    print("wrong recognized indeices are ", confuse)
    print("wrong recognized amount is ", len(confuse))
    print("wrong rate : ", 100*len(confuse)/len(labels), "%")

    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    #plt.title("Confusion picures #={}".format(len(confuse)))

    for i, idx in enumerate(confuse):
        plt.subplot(7, 7, 1+i)
        plt.imshow(pred_target[idx])
        plt.axis(False)
        plt.title("[{0}] p:{1}".format(idx, pred['class'][idx]))
    img_file_place = os.path.join(child_log_dir, "0_confusePics_{0:%y%m%d}_{1:%H%M}.png".format(now, now))
    plt.savefig(img_file_place)


if __name__ == '__main__':

    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)

    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    test_dir = os.path.join(data_dir, "test")
    print("test dir is in ... ", test_dir)
    #print(len(os.listdir(test_dir)))  # 2 (dog/cat)

    log_dir = os.path.join(cwd, "log")
    child_log_dir = os.path.join(log_dir, "da_source")  # ここは入力にしてもいいかもしれない。
    print(child_log_dir)
    print(os.listdir(child_log_dir))  # log list [history.pkl, model&weights.h5, log]

    
    main()
