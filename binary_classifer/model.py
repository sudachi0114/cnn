
# model を build するプログラム (共通)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

def build_model(INPUT_SIZE=224, CHANNEL=3):

    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNEL) 

    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=INPUT_SHAPE))
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

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])
    
    return model


if __name__ == '__main__':

    model = build_model()
    model.summary()
