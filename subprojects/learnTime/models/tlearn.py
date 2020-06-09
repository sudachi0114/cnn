#
# learn by transfer-learning model using imagenet weights
#

import os, sys
sys.path.append(os.pardir)

import time, datetime, gc
import numpy as np
# import pandas as pd

import tensorflow as tf
import keras
from keras import backend as K
print("TensorFlow version is ", tf.__version__)
print("Keras version is ", keras.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


def main(LEARN_PATH, IMAGE_SIZE, BATCH_SIZE, EPOCHS, FINE_TUNE_AT):
    """ function that train xception model

        # Args:
            LEARN_PATH (str): Path of dataset exists
            IMAGE_SIZE (int): The size of input images
            BATCH_SIZE (int): The number of images used as a batch for learning
            EPOCHS (int): The number of times the learning will do
            FINE_TUNE_AT (int): The number of layers 
                to unfreeze in fine-tune step of transfer-learning
        # Returns:
            None
                Logs the training of the model 
                and outputs the accuracy.
    """

    train_dir = os.path.join(LEARN_PATH, 'train')
    validation_dir = os.path.join(LEARN_PATH, 'validation')
    test_dir = os.path.join(LEARN_PATH, 'test')

    # calcucate the num of category
    num_category = 0
    for dirpath, dirnames, filenames in os.walk(train_dir):
        for dirname in dirnames:
            num_category += 1

    # All images will be resized to 299*299 or 598*598
    #   image_size valuable is argument of this function.
    #   Rescale all images by 1./255 and apply image augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode='categorical')

    # Create the base model from the pre-trained convnets
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)


    # Create the base model from the pre-trained model MobileNet V2
    """
    base_model = keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    """
    """
    base_model = keras.applications.mobilenet.MobileNet(input_shape=IMG_SHAPE,
                                                        include_top=False,
                                                        weights='imagenet')
    """
    """
    base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,
                                                             include_top=False,
                                                             weights='imagenet')
    """
    base_model = keras.applications.xception.Xception(input_shape=IMG_SHAPE,
                                                      include_top=False,
                                                      weights='imagenet')




    base_model.summary()
    # Freeze the convolutional base
    base_model.trainable = False

    # モデル
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_category, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # early stopping
    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       restore_best_weights=True)



    print("\ntraining sequence start .....")


    print("\nwarm up sequence .....")
    model.summary()

    # 更新される重みの数
    print('after', len(model.trainable_weights))

    # Train the model
    steps_per_epoch = train_generator.n // BATCH_SIZE
    validation_steps = validation_generator.n // BATCH_SIZE
    test_steps = test_generator.n // BATCH_SIZE

    start = time.time()
    _history = model.fit_generator(train_generator,
                                   steps_per_epoch = steps_per_epoch,
                                   epochs=EPOCHS,
                                   workers=4,
                                   validation_data=validation_generator,
                                   validation_steps=validation_steps,
                                   callbacks=[es],
                                   verbose=2)

    _val_pred = model.evaluate_generator(validation_generator,
                                         steps=validation_steps)
    print('val loss: {}, val acc: {}'.format(_val_pred[0], _val_pred[1]))


    # Fine tuning
    print("\nfine tuning.....")
    # Un-freeze the top layers of the model
    base_model.trainable = True

    # The nums of layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))


    # Freeze all the layers before the `FINE_TUNE_AT` layer
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    # Compile the model using a much-lower training rate
    model.compile(optimizer = keras.optimizers.Adam(lr=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


    # 更新される重みの数
    print('after Fine tune', len(model.trainable_weights))

    # Continue Train the model
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = steps_per_epoch,
                                  epochs=EPOCHS,
                                  workers=4,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  callbacks=[es],
                                  verbose=2)
    elapsed_time = time.time() - start
    print( "elapsed time (for train): {} [sec]".format(time.time() - start) )

    # print(history.history)
    model_val_acc = history.history['val_accuracy'][-1]
    print('val_acc: ', model_val_acc)

    # evaluate -----
    print("\nevaluate messages below ...")
    val_pred = model.evaluate_generator(validation_generator,
                                        steps=validation_steps)
    print('val loss: {}, val acc: {}'.format(val_pred[0], val_pred[1]))

    test_pred = model.evaluate_generator(test_generator,
                                         steps=test_steps)
    print('Test loss: {}, Test acc: {}'.format(test_pred[0], test_pred[1]))


    # confusion matrix & each class accuracy -----
    print("\nconfusion matrix")
    cm_pred = model.predict_generator(test_generator,
                                      steps=test_steps,
                                      verbose=2)

    test_label = []
    for i in range(test_steps):
        _, tmp_tl = next(test_generator)
        if i == 0:
            test_label = tmp_tl
        else:
            test_label = np.vstack((test_label, tmp_tl))

    idx_label = np.argmax(test_label, axis=-1)  # one_hot => normal
    idx_pred = np.argmax(cm_pred, axis=-1)  # 各 class の確率 => 最も高い値を持つ class

    cm = confusion_matrix(idx_label, idx_pred)
    print(cm)


if __name__ == '__main__':

    learn_path = "/home/sudachi/cnn/datasets/25000_721"

    main(LEARN_PATH=learn_path,
         IMAGE_SIZE=224,
         BATCH_SIZE=20,
         EPOCHS=50,
         FINE_TUNE_AT=108)
    # FINE TUNE AI:
    #   vgg16: 15
    #    mnv1: 81
    #    mnv2: 100?
    #    xception: 108
