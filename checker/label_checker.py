
# ImageDataGenerator するときの label を check するプログラム

import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def checker(dir_name, INPUT_SIZE=224, batch_size=10):

    TARGET_SIZE = (INPUT_SIZE, INPUT_SIZE)

    train_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(dir_name,
                                                        target_size=TARGET_SIZE,
                                                        batch_size=batch_size,
                                                        class_mode='binary')
    img_array, label = next(train_generator)
    print(img_array.shape, label.shape)

    plt.figure(figsize=(12, 6))

    for i in range(img_array.shape[0]):
        plt.subplot(2, 5, i+1)
        plt.imshow((img_array[i]))

        plt.title("label: {}".format(label[i]))

        plt.axis(False)

    plt.show()

if __name__ == '__main__':

    current_location = os.path.abspath(__file__)
    cwd, file_base = os.path.split(current_location)

    file_name, _ = os.path.splitext(file_base)

    cnn_dir = os.path.dirname(cwd)
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_mid300")
    train_dir = os.path.join(data_dir, "train")
    #validation_dir = os.path.join(data_dir, "validation")

    checker(train_dir)

    
