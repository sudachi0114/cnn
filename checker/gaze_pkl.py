
# pkl に保存したデータの検視

import os, pickle
import matplotlib.pyplot as plt


def load_pkl(target_file):

    with open(target_file, 'rb') as p:
        data_dict = pickle.load(p)

    print( data_dict.keys() )
    
    return data_dict


def display(data_dict):

    purpose_list = ['train', 'validation', 'test']
    purpose = purpose_list[0]

    data = data_dict['{}_data'.format(purpose)]
    label = data_dict['{}_label'.format(purpose)]

    print("data's shape: ", data.shape)
    print("label's shape: ", label.shape)

    data /= 255.0
    
    plt.figure(figsize=(12,6))

    for i in range(len(label)):
        plt.subplot(10, 10, i+1)
        plt.imshow(data[i])
        plt.title("class:{}".format(label[i]))
        plt.axis(False)

    plt.show()

if __name__ == '__main__':

    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    data_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller")

    print(data_dir, os.listdir(data_dir))

    target_file = os.path.join(data_dir, "data_dict.pkl")

    data_dict = load_pkl(target_file)
    display(data_dict)
