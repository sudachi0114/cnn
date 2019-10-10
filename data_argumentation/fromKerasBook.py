
# 学習データが少ない場合はデータに幾何的な変形を施して水増しする。

import os

import tensorflow as tf
# GPU を用いるときの tf の session の設定
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam


def main(input_size=150, batch_size=10, epochs=100):
    
    # directory -----
    current_location = os.path.abspath(__file__)
    cwd, base_name = os.path.split(current_location)
    file_name, _ = os.path.splitext(base_name)
    print("current location : ", cwd, ", this file : ", file_name)
    
    cnn_dir = os.path.dirname(cwd)

    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    train_dir = os.path.join(data_dir, "train")
    print("train data is in ... ", train_dir)
    validation_dir = os.path.join(data_dir, "validation")
    print("validation data is in ... ", validation_dir)

    # make log directory -----
    log_dir = os.path.join(cwd, "log")
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, "{}_log".format(file_name))
    os.makedirs(child_log_dir, exist_ok=True)
    


    # rescalling all image to 1/255, and padding train data
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    
    # we should not padding validation data
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(input_size, input_size),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=(input_size, input_size),
                                                                  batch_size=batch_size,
                                                                  class_mode='binary')

    # data shape
    data_checker, label_checker = next(train_generator)
    data_shape = data_checker.shape  # (batch_size, input_size, input_size, ch)
    print("data shape : ", data_shape)
    print("label shape : ", label_checker.shape)

    

    # model -----
    model = Sequential()

    # --- input ---
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(data_shape[1], data_shape[2], data_shape[3])))
    model.add(MaxPooling2D((2,2)))
    # ---
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    # ---
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    # ---
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    # ---
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # --- output ---

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])

    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size

    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " [steps (validation)]")

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps)

    # save model&weights in hdf5 file
    model.save(os.path.join(child_log_dir, '{}_model.h5'.format(file_name)))

    # save history
    import pickle
    with open(os.path.join(child_log_dir, '{}_history.pkl'.format(file_name)), 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)


if __name__ == '__main__':
    main()
