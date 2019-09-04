
# trial CNN in keras
#   CNN で MNIST

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

from keras.datasets import mnist
from keras.utils import to_categorical

import tensorflow as tf
# GPU を用いるときの tf の session の設定
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train.astype('float32') / 255.0

    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)


    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    # --- CNN --> flatten -----
    model.add(Flatten())
    # ----- flatten --> 全結合層 -----
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print("test loss : {0} | test accuracy : {1}".format(test_loss, test_acc))

    # save model in json file
    model2json = model.to_json()
    with open('cnn_minist_model.json', 'w') as f:
        f.write(model2json)

    # save weights in hdf5 file
    model.save_weights('cnn_minist_weights.h5')

    # save history
    import pickle
    with open('cnn_mnist_history.pkl', 'wb') as p:
        pickle.dump(history.history, p)

if __name__ == '__main__':
    main()


# モデルの保存に関して参考
#   https://qiita.com/ak11/items/67118e11b756b0ee83a5
#   https://m0t0k1ch1st0ry.com/blog/2016/07/17/keras/
#   https://keras.io/ja/getting-started/faq/#keras-model (公式)
