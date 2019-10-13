
# model を build するプログラム (共通)

from keras.models import Sequential
from keras.applications import VGG16
from keras.layers import Flatten, Dropout, Dense
from keras.optimizers import Adam

def build_model(INPUT_SIZE=224, CHANNEL=3):


    model = Sequential()
    
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNEL) 

    base_model = VGG16(input_shape=INPUT_SHAPE,
                       weights='imagenet',
                       include_top=False)

    #base_model.summary()

    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print("trainable weights before freeze: ", len(model.trainable_weights))

    # conv_base のパラメータを凍結
    base_model.trainable = False
    print("trainable weights after freeze: ", len(model.trainable_weights))


    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])
    
    return model


if __name__ == '__main__':

    model = build_model()
    model.summary()
