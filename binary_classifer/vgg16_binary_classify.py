
# pre-trained model : VGG16 を用いて画像認識 (特徴量抽出 feature extraction)

import os
import numpy as np

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.applications import VGG16
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def conv_gen(input_size, ch, summary=True):

    # using Conv base@VGG16
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(input_size, input_size, ch))

    if summary:
        conv_base.summary()

    return conv_base


def feature_extracter(conv_base, directory, input_size, data_size, batch_size=10):
    features = np.zeros(shape=(data_size, 4, 4, 512))
    labels = np.zeros(shape=(data_size))

    datagen = ImageDataGenerator(rescale=1/255.)

    generator = datagen.flow_from_directory(directory,
                                            target_size=(input_size, input_size),
                                            batch_size=batch_size,
                                            class_mode='binary')

    for i in range(generator.n // batch_size):
        inputs_batch, labels_batch = next(generator)
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size : (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch

    return features, labels


def main(input_size=150, ch=3, batch_size=10, epochs=30, train_size=100, validation_size=50):

    # directory -----
    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    base_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    train_dir = os.path.join(base_dir, "train")
    print("train data is in ... ", train_dir)
    test_dir = os.path.join(base_dir, "test")
    print("test data is in ... ", test_dir)

    log_dir = os.path.join(cnn_dir, "log")
    child_log_dir = os.path.join(log_dir, "vgg16_binary_classifer_log")
    os.makedirs(child_log_dir, exist_ok=True)

    # create conv_base -----
    conv_base = conv_gen(input_size, ch)

    # feature extraction -----
    train_features, train_labels = feature_extracter(conv_base, train_dir, input_size, train_size)
    validation_features, validation_labels = feature_extracter(conv_base, test_dir, input_size, validation_size)

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
    model.save(os.path.join(child_log_dir, "vgg16_binary_classify_model.h5"))

    # save history -----
    import pickle
    with open(os.path.join(child_log_dir, "vgg16_binary_classify_history.pkl"), 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)


if __name__ == '__main__':
    main()
