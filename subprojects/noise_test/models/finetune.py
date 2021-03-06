
# transfer learning with finetuning

import os, sys
sys.path.append(os.pardir)

import time
import numpy as np
np.random.seed(seed=114)

import tensorflow as tf
import keras
from keras import backend as K
config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.5
sess = tf.Session(config=config)
K.set_session(sess)

print("TensorFlow version is ", tf.__version__)
print("Keras version is ", keras.__version__)

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix

from utils.model_handler import ModelHandler


# define -----
batch_size = 50
input_size = 224
channel = 3
target_size = (input_size, input_size)
input_shpe = (input_size, input_size, channel)
set_epochs = 40



def main():

    cwd = os.getcwd()
    sub_prj = os.path.dirname(cwd)

    data_dir = os.path.join(sub_prj, "datasets")

    data_src = os.path.join(data_dir, "small_721")
    print("\ndata source: ", data_src)


    use_da_data = False
    if use_da_data:
        train_dir = os.path.join(data_src, "train_with_aug")
    else:
        train_dir = os.path.join(data_src, "train")
    validation_dir = os.path.join(data_src, "validation")
    test_dir = os.path.join(data_src, "test")

    print("train_dir: ", train_dir)
    print("validation_dir: ", validation_dir)
    # print("test_dir: ", test_dir)


    # data load ----------
    data_gen = ImageDataGenerator(rescale=1./255)

    train_generator = data_gen.flow_from_directory(train_dir,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   class_mode='categorical')
                                                           
    validation_generator = data_gen.flow_from_directory(validation_dir,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical')
                                                           
    test_generator = data_gen.flow_from_directory(test_dir,
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  class_mode='categorical')

    data_checker, label_checker = next(train_generator)

    print("train data shape (in batch): ", data_checker.shape)
    print("train label shape (in batch): ", label_checker.shape)
    # print("validation data shape:", validation_data.shape)
    # print("validation label shape:", validation_label.shape)
    # print("test data shape:", test_data.shape)
    # print("test label shape:", test_label.shape)
    

    # build model ----------
    mh = ModelHandler(input_size, channel)

    # あとで重みの解凍をできるように base_model を定義
    base_model = mh.buildMnv1Base()
    base_model.trainable=False

    model = mh.addChead(base_model)

    model.summary()


    # instance EarlyStopping -----
    es = EarlyStopping(monitor='val_loss',
                       patience=5,
                       verbose=1,
                       restore_best_weights=True)


    print("\ntraining sequence start .....")
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")                

    start = time.time()
    print("\ntraining sequence start .....")

    # 準備体操 -----
    print("\nwarm up sequence .....")
    model.summary()
    _history = model.fit_generator(train_generator,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=set_epochs,
                                   validation_data=validation_generator,
                                   validation_steps=validation_steps,
                                   callbacks=[es],
                                   verbose=1)

    # fine tuning -----
    print("\nfine tuning.....")
    mh.setFineTune(base_model, model, 81)
    model.summary()

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=set_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  callbacks=[es],
                                  verbose=1)

    elapsed_time = time.time() - start
    print( "elapsed time (for train): {} [sec]".format(time.time() - start) )


    print("\nevaluate sequence...")
    test_steps = test_generator.n // batch_size
    eval_res = model.evaluate_generator(test_generator,
                                        steps=test_steps,
                                        verbose=1)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])

    # confusion matrix -----
    print("\nconfusion matrix")
    pred = model.predict_generator(test_generator,
                                   steps=test_steps,
                                   verbose=3)

    test_label = []
    for i in range(test_steps):
        _, tmp_tl = next(test_generator)
        if i == 0:
            test_label = tmp_tl
        else:
            test_label = np.vstack((test_label, tmp_tl))    

    idx_label = np.argmax(test_label, axis=-1)  # one_hot => normal
    idx_pred = np.argmax(pred, axis=-1)  # 各 class の確率 => 最も高い値を持つ class
    
    cm = confusion_matrix(idx_label, idx_pred)

    # Calculate Precision and Recall
    tn, fp, fn, tp = cm.ravel()


    print("  | T  | F ")
    print("--+----+---")
    print("N | {} | {}".format(tn, fn))
    print("--+----+---")
    print("P | {} | {}".format(tp, fp))

    # 適合率 (precision):
    # precision = tp/(tp+fp)
    # print("Precision of the model is {}".format(precision))

    # 再現率 (recall):
    # recall = tp/(tp+fn)
    # print("Recall of the model is {}".format(recall))


    # save model -----
    save_location = os.path.join(sub_prj, "outputs", "models")
    if use_da_data == True:
        save_file = os.path.join(save_location, "auged_finetune_model.h5")
    else:
        save_file = os.path.join(save_location, "finetune_model.h5")
    model.save(save_file)
    print("\nmodel has saved in", save_file)



if __name__ == '__main__':
    main()
