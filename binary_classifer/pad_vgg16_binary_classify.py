
# 水増しデータ + 転移学習 (VGG16)
import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam



def main(input_size=150, batch_size=10, epochs=30):

    # directory ----
    current_location = os.path.abspath(__file__)
    cwd, base_name = os.path.split(current_location)
    file_name, _ = os.path.splitext(base_name)
    print("current location : ", cwd, ", this file : ", file_name)

    cnn_dir = os.path.dirname(cwd)

    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")
    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "validation")

    print("train datas are in ... ", train_dir)
    print("validation datas are in ... ", validation_dir)

    log_dir = os.path.join(cwd, "log")
    child_log_dir = os.path.join(log_dir, "{}_log".format(file_name))
    os.makedirs(child_log_dir, exist_ok=True)

    # data gen -----
    train_datagen = ImageDataGenerator(rescale=1/255.0,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    validation_datagen = ImageDataGenerator(rescale=1/255.0)


    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(input_size, input_size),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=(input_size, input_size),
                                                                  batch_size=batch_size,
                                                                  class_mode='binary')

    data_checker, label_checker = next(train_generator)
    data_shape = data_checker.shape

    print("train data shape : ", data_shape)
    print("validation data shape : ", label_checker.shape)    

    # conv_base -----
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(data_shape[1], data_shape[2], data_shape[3]))
                          
    
    # model -----
    model = Sequential()

    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    print("trainable params before freezing conv_base", len(model.trainable_weights))

    # set trainable=False @conv_base param
    conv_base.trainable = False
    
    print("trainable params after freezing conv_base", len(model.trainable_weights))

    # compile ---
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])

    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size

    # fit -----
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  verbose=1)

    # save model -----
    model.save(os.path.join(child_log_dir, "{}_model.h5".format(file_name)))

    # save history -----
    import pickle
    with open(os.path.join(child_log_dir, "{}_history.pkl".format(file_name)), 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)

if __name__ == '__main__':
    main()
