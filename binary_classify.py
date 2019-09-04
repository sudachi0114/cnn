
# 2値分類 のプログラムにする。

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from keras.optimizers import Adam

def main():

    input_size = 150

    #if x_train.shape[3] == 3:
    if 2 == 3:
        ch = 3
    else:
        ch = 1
    
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
                  mertrics=['accuracy'])


if __name__ == '__main__':
    main()
