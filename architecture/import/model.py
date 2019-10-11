
# よりプログラムを綺麗に書くための設計
#   import を使ってファイルを分割する編

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import MobileNetV2


def create_model(INPUT_SIZE=224, CHANNEL=3):
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNEL)

    base_model = MobileNetV2(input_shape=INPUT_SHAPE,
                             weights='imagenet',
                             include_top=False)
    model = Sequential()
    # 設計は沈さんのモデルを拝借 (activites/20190718)
    model.add(base_model)
    #model.add(GlobalAveragePooling2D())
    # activites/20190712 のものも採用
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 2値分類の時は sigmoid を選択する

    model.compile(optimizer=Adam(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


        

if __name__ == '__main__':

    model = create_model()
