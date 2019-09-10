
# hisotry から accuracy と loss をプロット

import os, pickle

import matplotlib.pyplot as plt

cwd = os.getcwd()
cnn_dir = os.path.dirname(cwd)
print(cnn_dir)

binary_path = os.path.join(cnn_dir, "binary_classifer")

file_name = "binary_dogs_vs_cats_history.pkl"

target = os.path.join(binary_path, file_name)

with open(target, mode='rb') as pkl:
    history = pickle.load(pkl)

    print(history.keys())  # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

    acc = history['acc']
    #print(acc)
    val_acc = history['val_acc']

    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc)+1)

    # acc / val_acc を plt
    plt.plot(epochs, acc, 'm', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.figure()

    # loss / val_loss を plt
    plt.plot(epochs, loss, 'm', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()
