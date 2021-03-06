
import os, sys
sys.path.append(os.pardir)

import time, datetime, gc
import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.5
sess = tf.Session(config=config)
K.set_session(sess)

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from utils.model_handler import ModelHandler
# from utils.img_utils import inputDataCreator, dataSplit

# define -----
batch_size = 50
input_size = 224
channel = 3
target_size = (input_size, input_size)
input_shape = (input_size, input_size, channel)
set_epochs = 40



def main():

    cwd = os.getcwd()
    sub_prj = os.path.dirname(cwd)
    sub_prj_root = os.path.dirname(sub_prj)
    prj_root = os.path.dirname(sub_prj_root)

    data_dir = os.path.join(prj_root, "datasets")

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
    print("test_dir: ", test_dir)


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
    model = mh.buildMyModel()
    model.summary()


    # instance EarlyStopping -----
    es = EarlyStopping(monitor='val_loss',
                       # monitor='val_accuracy',
                       patience=5,
                       verbose=1,
                       restore_best_weights=True)



    print("\ntraining sequence start .....")
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")                

    start = time.time()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=set_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  callbacks=[es],
                                  verbose=1)
    elapsed_time = time.time() - start
    print( "elapsed time (for train): {} [sec]".format(elapsed_time) )


    # evaluate ----------
    print("\nevaluate sequence...")

    accs = history.history['accuracy']
    losses = history.history['loss']
    val_accs = history.history['val_accuracy']
    val_losses = history.history['val_loss']
    print("last val_acc: ", val_accs[len(val_accs)-1])
    
    test_steps = test_generator.n // batch_size
    eval_res = model.evaluate_generator(test_generator,
                                        steps=test_steps,
                                        verbose=1)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])

    # logging and detail outputs -----
    # make log_dirctory
    log_dir = os.path.join(sub_prj, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    model_log_dir = os.path.join(sub_prj, "outputs", "models")
    os.makedirs(log_dir, exist_ok=True)

    now = datetime.datetime.now()
    child_log_dir = os.path.join(log_dir, "{0:%Y%m%d}".format(now))
    os.makedirs(child_log_dir, exist_ok=True)
    child_model_log_dir = os.path.join(model_log_dir, "{0:%Y%m%d}".format(now))
    os.makedirs(child_model_log_dir, exist_ok=True)

    # save model & weights
    model_file = os.path.join(child_model_log_dir, "model.h5")
    model.save(model_file)
    print("\nexport model in ", child_model_log_dir)


    print("\npredict sequence...")
    pred = model.predict_generator(test_generator,
                                   steps=test_steps,
                                   verbose=1)

    test_label = []
    for i in range(test_steps):
        _, tmp_tl = next(test_generator)
        if i == 0:
            test_label = tmp_tl
        else:
            test_label = np.vstack((test_label, tmp_tl))    

    label_name_list = []
    for i in range(len(test_label)):
        if test_label[i][0] == 1:
            label_name_list.append('cat')
        elif test_label[i][1] == 1:
            label_name_list.append('dog')
        

    #print("result: ", pred)
    df_pred = pd.DataFrame(pred, columns=['cat', 'dog'])
    df_pred['class'] = df_pred.idxmax(axis=1)
    df_pred['label'] = pd.DataFrame(label_name_list, columns=['label'])
    df_pred['collect'] = (df_pred['class'] == df_pred['label'])

    confuse = df_pred[df_pred['collect'] == False].index.tolist()
    collect = df_pred[df_pred['collect'] == True].index.tolist()

    print(df_pred)
    print("\nwrong recognized indeices are ", confuse)
    print("  wrong recognized amount is ", len(confuse))
    print("\ncollect recognized indeices are ", collect)
    print("  collect recognized amount is ", len(collect))
    print("\nwrong rate: ", 100*len(confuse)/len(test_label), " %")


    # save history
    save_dict = {}
    save_dict['last_loss'] = losses[len(losses)-1]
    save_dict['last_acc'] = accs[len(accs)-1]
    save_dict['last_val_loss'] = val_losses[len(val_losses)-1]
    save_dict['last_val_acc'] = val_accs[len(val_accs)-1]
    save_dict['n_confuse'] = len(confuse)
    save_dict['eval_loss'] = eval_res[0]
    save_dict['eval_acc'] = eval_res[1]
    save_dict['elapsed_time'] = elapsed_time

    print(save_dict)

    df_result = pd.DataFrame(save_dict.values(), index=save_dict.keys())

    csv_file = os.path.join( child_log_dir, "result.csv" )
    df_result.to_csv(csv_file)
    print("\nexport history in ", csv_file)


if __name__ == '__main__':
    main()
