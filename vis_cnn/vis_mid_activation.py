
# CNN の中間層の出力を可視化する。
#   model と weight が一緒の h5 file に保存されている必要がありそう..

import os

from keras.models import load_model

def main():

    # directory -----
    cwd = os.getcwd()
    cnn_dir = os.path.dirname(cwd)
    log_dir = os.path.join(cnn_dir, "log")

    log_list = os.listdir(log_dir)
    print("\nplease chose log below : \n", log_list)
    choice = input(">>> ")

    child_log_dir = os.path.join(log_dir, choice)
    print("You choose", child_log_dir, "\n")

    child_log_list = os.listdir(child_log_dir)
    print("\nplease chose saved model below : \n", child_log_list)
    saved_model = input(">>> ")

    model_location = os.path.join(child_log_dir, saved_model)
    print("Your model is ", model_location, "\n")

    
    # load model -----
    model = load_model(model_location)

    model.summary()


if __name__ == '__main__':
    main()
