
# pre-trained model : VGG16 を用いて画像認識 (特徴量抽出 feature extraction)
#   特徴量抽出を先に行い、その結果を用いて、NNで学習する方法。

import os, sys
sys.path.append(os.pardir)
import numpy as np

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from data_handler import DataHandler
from model_handler import ModelHandler


def feature_extracter(conv_base, directory, input_size, data_size, batch_size=10):
    features = np.zeros(shape=(data_size, 4, 4, 512))
    labels = np.zeros(shape=(data_size))

    data_handler = DataHandler()
    print("INPUT_SIZE before: ", data_handler.INPUT_SIZE)
    data_handler.INPUT_SIZE = input_size
    print("INPUT_SIZE after: ", data_handler.INPUT_SIZE)

    generator = data_handler.dataGenerator(directory)

    for i in range(generator.n // batch_size):
        inputs_batch, labels_batch = next(generator)
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size : (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch

    return features, labels


def main(input_size=150, ch=3, batch_size=10, epochs=30, train_size=100, validation_size=50):

    # directory -----
    current_location = os.path.abspath(__file__)
    cwd, base_name = os.path.split(current_location)
    file_name, _ = os.path.splitext(base_name)
    print("current location : ", cwd, ", this file : ", file_name)
    
    cnn_dir = os.path.dirname(cwd)
    base_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    train_dir = os.path.join(base_dir, "train")
    print("train data is in ... ", train_dir)
    validation_dir = os.path.join(base_dir, "validation")
    print("validation data is in ... ", validation_dir)

    log_dir = os.path.join(cwd, "log")
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, "{}_log".format(file_name))
    os.makedirs(child_log_dir, exist_ok=True)

    # create conv_base -----
    model_handler = ModelHandler(input_size, ch)
    conv_base = model_handler.buildVgg16Base()

    # feature extraction -----
    train_features, train_labels = feature_extracter(conv_base, train_dir, input_size, train_size)
    validation_features, validation_labels = feature_extracter(conv_base, validation_dir, input_size, validation_size)

    # reshape (Flatten) -----
    train_features = np.reshape(train_features, (train_size, 4*4*512))
    validation_features = np.reshape(validation_features, (validation_size, 4*4*512))
    

    # model -----
    model = Sequential()

    model.add(Dense(256, activation='relu', input_dim=4*4*512))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])

    history = model.fit(train_features, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_features, validation_labels))

    # save model in hdf5 file -----
    model.save(os.path.join(child_log_dir, "{}_model.h5".format(file_name)))

    # save history -----
    import pickle
    with open(os.path.join(child_log_dir, "{}_history.pkl".format(file_name)), 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)


if __name__ == '__main__':
    main()
