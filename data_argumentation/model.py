
# DA 評価用 共通モデル

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.applications import MobileNetV2

def build_model(INPUT_SIZE=224, CHANNEL=3, mode='fe'):

    INPUT_SHAPE= (INPUT_SIZE, INPUT_SIZE, CHANNEL)

    base_model = MobileNetV2(input_shape=INPUT_SHAPE,
                             weights='imagenet',
                             include_top=False)
    #base_model.summary()

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #model.summary()

    if mode == 'fe':  # [f]eature [e]xtraction
        base_model.trainable = False
        print("trainable params: ", len(model.trainable_weights))
    elif mode == 'ft':  # [f]ine [t]uning
        base_model.trainable = True
        print("trainable params before freeze: ", len(model.trainable_weights))
        is_trainable = False
        for layer in base_model.layers:
            if layer.name == 'block_8_expand':
                is_trainable = True
            if is_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        print("trainable params >after< freeze: ", len(model.trainable_weights))

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    return model


if __name__ == '__main__':

    model = build_model()
    model.summary()
