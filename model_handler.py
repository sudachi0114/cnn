
import os

from keras.models import Sequential
from keras.applications import VGG16, MobileNetV2
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

class BuildModel:

    def __init__(self):
        print("instanced")

        self.INPUT_SIZE = 224
        self.CHANNEL = 3
        self.INPUT_SHAPE = (self.INPUT_SIZE, self.INPUT_SIZE, self.CHANNEL)

    def buildMyModel(self):

        model = Sequential()

        model.add(Conv2D(32, (3,3), activation='relu', input_shape=self.INPUT_SHAPE))
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

    
    def buildVgg16Base(self):

        base_model = VGG16(input_shape=self.INPUT_SHAPE,
                           weights='imagenet',
                           include_top=False)

        return base_model

    
    def buildMnv2Base(self):

        base_model = MobileNetV2(input_shape=self.INPUT_SHAPE,
                                 weights='imagenet',
                                 include_top=False)

        return base_model


    # 転移 base に + 分類 head する
    def buildTlearnModel(self, base='vgg16'):

        model = Sequential()

        if base == 'vgg16':
            base_model = self.buildVgg16Base()
        elif base == 'mnv2':
            base_model = self.buildMnv2Base()

        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        return model

        

        

if __name__ == '__main__':

    model_hander = BuildModel()

    #model = model_hander.buildMyModel()
    #model = model_hander.buildVgg16Model()
    model = model_hander.buildTlearnModel(base='vgg16')
    
    model.summary()
