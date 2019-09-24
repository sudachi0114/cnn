
# hisotry から accuracy と loss をプロット

import os, pickle

import matplotlib.pyplot as plt

cwd = os.getcwd()
cnn_dir = os.path.dirname(cwd)
log_dir = os.path.join(cnn_dir, "log")
print("logs are in ", log_dir)

log_list = os.listdir(log_dir)
print("please chose folder from ...\n", log_list)
choice_log = input(">>> ")

child_log_dir = os.path.join(log_dir, choice_log)

file_list = os.listdir(child_log_dir)

found = 0
target_list = []

for file_name in file_list:
    if file_name.find("history") != -1:
        found += 1
        print("find {} history file!".format(found))
        target_list.append(file_name)
        print(target_list)
    else:
        pass

if not found:
    print("please chose folder from ...\n", file_list)
    file_name = input(">>> ")    
elif found > 1:
    print("please chose folder from ...\n", target_list)
    file_name = input(">>> ")
elif found == 1:
    file_name = target_list[0]


file_path = os.path.join(child_log_dir, file_name)
print("open file in ", file_path)

with open(file_path, mode='rb') as pkl:
    history = pickle.load(pkl)

    keys = history.keys()
    print("dict keys : ", keys)  # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
    # acc の場合と accuray の場合がある。(おそらく metrices で指定した値によって変わる)

    acc = history['accuracy']
    #print(acc)
    val_acc = history['val_accuracy']

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
