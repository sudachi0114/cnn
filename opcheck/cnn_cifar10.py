
# trial CNN in keras
#   CNN で CIFAR10

# ----- 学習の実行時間を計測する -----
import time, os

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

from keras.datasets import cifar10
from keras.utils import to_categorical

import tensorflow as tf
# GPU を用いるときの tf の session の設定
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def main(rgb=True):
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    input_size = x_train.shape[1]

    if rgb:
        ch = 3
    else:
        ch = 1


    x_train = x_train.reshape((50000, input_size, input_size, ch))
    x_train = x_train.astype('float32') / 255.0

    x_test = x_test.reshape((10000, input_size, input_size, ch))
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)


    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(input_size,input_size,ch)))
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

    # stop watch start!
    start = time.time()
    
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1)

    # stop watch stop!
    elapsed_time = time.time() - start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print("test loss : {0} | test accuracy : {1}".format(test_loss, test_acc))

    # print elapsed time
    print("train takes {} [sec]".format(elapsed_time))

    # directory ---
    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    log_dir = os.path.join(cnn_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    child_log_dir = os.path.join(log_dir, "opcheck_cifar10_log")
    os.makedirs(child_log_dir, exist_ok=True)

    # save model & weights -----
    model.save(os.path.join(child_log_dir, 'cnn_cifar10_model.h5'))

    # save history
    import pickle
    with open(os.path.join(child_log_dir, 'cnn_cifar10_history.pkl'), 'wb') as p:
        pickle.dump(history.history, p)

    print("export log in ", child_log_dir)

if __name__ == '__main__':
    main()

