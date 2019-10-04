
# Data Argumentation の効果検証用プログラム

import os

import tensorflow as tf
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
tf.Session(config=session_config)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# 共通
def validationDataGenerator(validation_dir, input_size=150, batch_size=10):

    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)    
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=(input_size, input_size),
                                                                  batch_size=batch_size,
                                                                  class_mode='binary')
    return validation_generator

# これをいじっていく
def trainDataGenerator(train_dir, input_size=150, batch_size=10):

    train_datagen = ImageDataGenerator(rescale=1.0/255.0)  # 何もしない (rescale だけ)
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(input_size, input_size),
                                                        batch_size=batch_size,
                                                        class_mode='binary')
    return train_generator



def create_model(input_size=150, ch=3):

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

    return model



def main(input_size=150, batch_size=10):

    validation_generator = validationDataGenerator(validation_dir)
    train_generator = trainDataGenerator(validation_dir)

 
    data_checker, label_checker = next(train_generator)
    print("data shape : ", data_checker.shape)
    print("label shape : ", label_checker.shape)

    #print(data_checker[1])
    #print(label_checker[1])

    ch = data_checker.shape[3]
    print("set channel : ", ch)
    print("batch_size : ", batch_size)

    model = create_model()

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
                                  verbose=1)

    # save model & weights
    model.save(os.path.join(child_log_dir, 'da_source_model.h5'))

    # save history
    import pickle
    with open(os.path.join(child_log_dir, 'da_source_history.pkl'), 'wb') as p:
        pickle.dump(history.history, p)

    print("export logs in ", child_log_dir)


    
if __name__ == '__main__':

    cwd = os.getcwd()
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
    child_log_dir = os.path.join(log_dir, "da_source")
    os.makedirs(child_log_dir, exist_ok=True)

    
    main()
