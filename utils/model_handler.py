
import os

from keras.models import Sequential
from keras.applications import VGG16, MobileNet, MobileNetV2, Xception
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam

class ModelHandler:

    def __init__(self, input_size=224, channel=3):

        print("Model Handler has instanced.")

        self.INPUT_SIZE = input_size
        self.CHANNEL = channel
        self.INPUT_SHAPE = (self.INPUT_SIZE, self.INPUT_SIZE, self.CHANNEL)  # ch_last

        self.BASE_MODEL_FREEZE = True

        self.CLASS_MODE = 'categorical'  # 'binary'
        print("class mode:", self.CLASS_MODE)


    def modelCompile(self, model):

        if self.CLASS_MODE == 'categorical':
            adapt_loss = 'categorical_crossentropy'
        elif self.CLASS_MODE == 'binary':
            adapt_loss = 'binary_crossentropy',

        model.compile(loss=adapt_loss,
                      optimizer=Adam(lr=1e-4),
                      metrics=['accuracy'])

        print( "your model has compiled.")

        return model


    def buildMyModel(self):

        print("build simple model...")

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

        if self.CLASS_MODE == 'categorical':
            model.add(Dense(2, activation='softmax'))
        elif self.CLASS_MODE == 'binary':
            model.add(Dense(1, activation='sigmoid'))

        return self.modelCompile(model)

    
    def buildVgg16Base(self):

        print("building vgg16 base...")

        #self.INPUT_SIZE = 224  # default値

        base_model = VGG16(input_shape=self.INPUT_SHAPE,
                           weights='imagenet',
                           include_top=False)

        return base_model


    def buildMnv1Base(self):

        print("building MobileNetV1 base model...")

        #self.INPUT_SIZE = 224  # default値
        
        base_model = MobileNet(input_shape=self.INPUT_SHAPE,
                               weights='imagenet',
                               include_top=False)

        return base_model


    def buildMnv2Base(self):

        print("building MobileNetV2 base model...")

        #self.INPUT_SIZE = 224  # default値
        
        base_model = MobileNetV2(input_shape=self.INPUT_SHAPE,
                                 weights='imagenet',
                                 include_top=False)

        return base_model

    def buildXceptionBase(self):

        print("building Xception base model...")

        #self.INPUT_SIZE = 299  # default値

        base_model = Xception(input_shape=self.INPUT_SHAPE,
                              weights='imagenet',
                              include_top=False)

        return base_model


    # 転移 base に + 分類 head する
    def buildTlearnModel(self, base='vgg16'):
        
        model = Sequential()
        
        print("base model is {}".format(base))

        if base == 'vgg16':
            base_model = self.buildVgg16Base()
        elif base == 'mnv1':
            base_model = self.buildMnv1Base()
        elif base == 'mnv2':
            base_model = self.buildMnv2Base()
        elif base == 'xception':
            base_model = self.buildXceptionBase()

        print("attempt classifer head on base model.")

        model.add(base_model)

        if self.CLASS_MODE == 'categorical':
            model.add(GlobalAveragePooling2D())
            model.add(Dense(2, activation='softmax'))
        elif self.CLASS_MODE == 'binary':
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))  # base_model に寄らない設計でいいのか??
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

        # base_model のパラメータを凍結
        if self.BASE_MODEL_FREEZE:
            print("trainable weights before freeze: ", len(model.trainable_weights))
            base_model.trainable = False
            print("trainable weights after freeze: ", len(model.trainable_weights))


        return self.modelCompile(model)


if __name__ == '__main__':

    mh = ModelHandler()

    #model = mh.buildMyModel()
    #model = mh.buildVgg16Base()
    model = mh.buildTlearnModel(base='mnv1')

    model.summary()
