
# 2値分類 のプログラムにする。

# ----- import -----
import os

import tensorflow as tf
# GPU を用いるときの tf の session の設定
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

def main():

    input_size = 150
    batch_size = 32

    # directory define
    cwd = os.getcwd()
    #print("current : ", cwd)
    base_dir = os.path.join(cwd, "dogs_vs_cats_smaller")

    #cnn_dir = os.path.dirname(cwd)
    #base_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    train_dir = os.path.join(base_dir, "train")
    print("train data is in ... ", train_dir)
    test_dir = os.path.join(base_dir, "test")
    print("test data is in ... ", test_dir)
    
    # rescaring all images to 1/255
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_gen = train_datagen.flow_from_directory(train_dir,
                                                  target_size=(input_size, input_size),
                                                  batch_size=batch_size,
                                                  class_mode='binary')

    test_gen = test_datagen.flow_from_directory(test_dir,
                                                target_size=(input_size, input_size),
                                                batch_size=batch_size,
                                                class_mode='binary')
    data_checker, label_checker = next(train_gen)
    print("data shape : ", data_checker.shape)
    print("label shape : ", label_checker.shape)

    #print(data_checker[1])
    #print(label_checker[1])
    
    if data_checker.shape[3] == 3:
        ch = 3
    else:
        ch = 1
    print("set ch : ", ch)
        
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(input_size, input_size, ch)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])

    steps_per_epoch = train_gen.n // batch_size
    validation_steps = test_gen.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps < default=10)")

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=30,
                                  validation_data=test_gen,
                                  validation_steps=10)
    
    # save model in json file
    model2json = model.to_json()
    with open('binary_dogs_vs_cats_model.json', 'w') as f:
        f.write(model2json)

    # save weights in hdf5 file
    model.save_weights('binary_dogs_vs_cats_weights.h5')

    # save history
    import pickle
    with open('binary_dogs_vs_cats_history.pkl', 'wb') as p:
        pickle.dump(history.history, p)
                                  


if __name__ == '__main__':
    main()
