
# mode の全てに対して実行を行うプログラム

import os, sys
cwd = os.getcwd()
cnn_dir = os.path.dirname(cwd)

log_dir = os.path.join(cwd, "log")
os.makedirs(log_dir, exist_ok=True)

sys.path.append(os.pardir)

# -----
from da_handler import DaHandler
dah = DaHandler()

aug_list = dah.imgaug_mode_list
aug_list.pop(0)

# -----
from data_handler import DataHandler
dth = DataHandler()

# -----
from model_handler import ModelHandler
mh = ModelHandler(224, 3)  # input_size=224, ch=3(共通)

model = mh.buildMyModel()
model.summary()
# -----
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)


# -----
def data_create():

    print("Do while below -----\n", aug_list)

    for aug in aug_list:

        print("\nSaving {", aug, "} images...")
        dh.save_imgauged_img(mode='image', aug=aug)

        print("Done.")
# -----

def validation_data_generator():

    validation_dir = os.path.join(cnn_dir, "dogs_vs_cats_smaller", "validation")
    data_generator = dth.dataGenerator(validation_dir)

    return data_generator


def auged_data_generator():

    for aug in aug_list:
        auged_data_dir = os.path.join(cnn_dir, "dogs_vs_cats_auged_{}".format(aug))

        print(auged_data_dir)
        data_generator = dth.dataGenerator(auged_data_dir)

        yield data_generator


def train(trainDataGeneratorIterator, validation_generator):

    for aug in aug_list:
        print("\n========== process", aug, "phase ==========")
        # loop によって変わるものは 2つ
        #   1. train_generator
        #   2. aug (augmentation の内容と名前)
        train_generator = next(trainDataGeneratorIterator)

        data_checker, label_checker = next(train_generator)
        print("data_checker.shape: ", data_checker.shape)
        print("label_checker.shape: ", label_checker.shape)
        

        batch_size = data_checker.shape[0] # 10枚で共通なんだけどね。

        steps_per_epoch = train_generator.n // batch_size
        validation_steps = validation_generator.n // batch_size
        print(steps_per_epoch, " [steps / epoch]")
        print(validation_steps, " (validation steps)")

        # model (global に定義してある (ズル))
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=50,
                                      validation_data=validation_generator,
                                      validation_steps=validation_steps,
                                      verbose=1)

        # chidl_log_dir -----
        child_log_dir = os.path.join(log_dir, aug)
        os.makedirs(child_log_dir, exist_ok=True)

        # save model & weights
        model_file = os.path.join(child_log_dir, '{}_model.h5'.format(aug))
        model.save(model_file)

        # save history
        history_file = os.path.join(child_log_dir, '{}_history.pkl'.format(aug))
        with open(history_file, 'wb') as p:
            pickle.dump(history.history, p)

        print("export logs in ", child_log_dir)


if __name__ == '__main__':

    #data_create()
    validation_generator = validation_data_generator()
    #trainDataGenerator = next(auged_data_generator())  # generator を generate する..

    trainDataGeneratorIterator = auged_data_generator()

    train(trainDataGeneratorIterator, validation_generator)
