
# 2値分類 のプログラムにする。
#   dogs vs cats 12500*2 枚 全てを学習

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

    cnn_dir = os.path.dirname(cwd)
    base_dir = os.path.join(cnn_dir, "dogs_vs_cats_full")
    train_dir = os.path.join(base_dir, "train")
    print("train data is in ... ", train_dir)
    validation_dir = os.path.join(base_dir, "validation")
    print("validation data is in ... ", validation_dir)
    
    # rescaring all images to 1/255
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                  target_size=(input_size, input_size),
                                                  batch_size=batch_size,
                                                  class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=(input_size, input_size),
                                                                  batch_size=batch_size,
                                                                  class_mode='binary')
    
    data_checker, label_checker = next(train_generator)
    print("data shape : ", data_checker.shape)
    print("label shape : ", label_checker.shape)

    #print(data_checker[1])
    #print(label_checker[1])
    
    ch = data_checker.shape[3]
    print("set channel : ", ch)
        
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

    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=30,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  verbose=2
    )

    # make log dir
    log_dir = os.path.join(cnn_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, "binary_classifer_full_learn_log")
    os.makedirs(child_log_dir, exist_ok=True)
    
    # save model & weights
    model.save(os.path.join(child_log_dir, 'binary_dogs_vs_cats_full_learn_model.h5'))

    # save history
    import pickle
    with open(os.path.join(child_log_dir, 'binary_dogs_vs_cats_full_learn_history.pkl'), 'wb') as p:
        pickle.dump(history.history, p)
                                  
    print("export logs in ", child_log_dir)

if __name__ == '__main__':
    main()
