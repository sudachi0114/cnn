
import os, sys, argparse, pickle, csv, time
sys.path.append(os.pardir)

import numpy as np
import pandas as pd

import tensorflow as tf
import keras
import gc
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from utils.img_utils import inputDataCreator, dataSplit
from utils.model_handler import ModelHandler


cwd = os.getcwd()


def main(data_mode, model_mode, no, set_epochs=60, do_es=False):

    batch_size = 10

    if data_mode == 'native':
        data_dir = os.path.join(cwd, "experiment_{}".format(no))

        total_data, total_label = inputDataCreator(data_dir,
                                                   224,
                                                   normalize=True,
                                                   one_hot=True)

        train_data, train_label, validation_data, validation_label, test_data, test_label = dataSplit(total_data, total_label)

        """
        print("\ntrain data shape: ", train_data.shape)
        print("train label shape: ", train_label.shape)
        print("\nvalidation data shape: ", validation_data.shape)
        print("validation label shape: ", validation_label.shape)
        print("\ntest data shape: ", test_data.shape)
        print("test label shape: ", test_label.shape)
        """

        datagen = ImageDataGenerator()

        train_generator = datagen.flow_from_directory(train_dir,
                                                      target_size=(224, 224),
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      class_mode='categorical')
        validation_generator = datagen.flow_from_directory(validation_dir,
                                                           target_size=(224, 224),
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           class_mode='categorical')
        test_generator = datagen.flow_from_directory(test_dir,
                                                     target_size=(224, 224),
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     class_mode='categorical')


    elif data_mode == 'auged':
        set_epochs = int( set_epochs/2 )
        data_dir = os.path.join(cwd, "concat_experiment_{}".format(no))

        train_dir = os.path.join(data_dir, "train")
        validation_dir = os.path.join(data_dir, "validation")
        test_dir = os.path.join(data_dir, "test")

        datagen = ImageDataGenerator(rescale=1/255.0)

        train_generator = datagen.flow_from_directory(train_dir,
                                                      target_size=(224, 224),
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      class_mode='categorical')
        validation_generator = datagen.flow_from_directory(validation_dir,
                                                           target_size=(224, 224),
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           class_mode='categorical')
        test_generator = datagen.flow_from_directory(test_dir,
                                                     target_size=(224, 224),
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     class_mode='categorical')



    data_sample, label_sample = next(train_generator)
    input_size = data_sample.shape[1]
    channel = data_sample.shape[3]
    print( "input_size: {} | channel: {}".format(input_size, channel) )
    print("set epochs: ", set_epochs)


    mh = ModelHandler(input_size, channel)

    if model_mode == 'mymodel':
        model = mh.buildMyModel()
    elif model_mode == 'tlearn':
        model = mh.buildTlearnModel(base='mnv1')

    model.summary()

    if do_es:
        es = EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           mode='auto')
        es = [es]
    else:
        es = None

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
                                  verbose=1)

                                  

    elapsed_time = time.time() - start

    accs = history.history['accuracy']
    losses = history.history['loss']
    val_accs = history.history['val_accuracy']
    val_losses = history.history['val_loss']


    if do_es:
        log_dir = os.path.join(cwd, "log_with_es")
    else:
        log_dir = os.path.join(cwd, "log")
    os.makedirs(log_dir, exist_ok=True)

    """
    child_log_dir = os.path.join(log_dir, "{}_{}_{}".format(data_mode, model_mode, no))
    os.makedirs(child_log_dir, exist_ok=True)

    # save model & weights
    model_file = os.path.join(child_log_dir, "{}_{}_{}_model.h5".format(data_mode, model_mode, no))
    model.save(model_file)

    # save history
    history_file = os.path.join(child_log_dir, "{}_{}_{}_history.pkl".format(data_mode, model_mode, no))
    with open(history_file, 'wb') as p:
        pickle.dump(history.history, p)

    print("\nexport logs in ", child_log_dir)
    """



    print("\npredict sequence...")


    pred = model.predict_generator(test_generator,
                                   verbose=1)


    label_name_list = []

    if data_mode == 'auged':
        test_steps = test_generator.n // batch_size
        for i in range(test_steps):
            _, tmp = next(test_generator)
            if i == 0:
                test_label = tmp
            else:
                test_label = np.vstack((test_label, tmp))

    for i in range(len(test_label)):
        if test_label[i][0] == 1:
            label_name_list.append('cat')
        elif test_label[i][1] == 1:
            label_name_list.append('dog')

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


    print("\nevaluate sequence...")


    eval_res = model.evaluate(test_generator,
                              verbose=1)
    

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])

    # ----------
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

    # undefine validable ----------
    del datagen
    del train_generator, validation_generator, test_generator
    
    #   due to CPU memory ---------
    # del train_data, train_label
    # del validation_data, validation_label
    # del test_data, test_label
    # del set_epochs


    #   due to GPU memory ---------
    # del mh,
    del model
    del history
    # del accs, losses, val_accs, val_losses
    # del pred, df_pred, label_name_list
    # del confuse, collect
    # del eval_res
    

    keras.backend.clear_session()
    gc.collect()

    return save_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DA 実験 100例 学習プログラム")

    parser.add_argument("--earlystopping", "-es", action="store_true",
                        help="学習時に EarlyStopping を ON にする.")

    args = parser.parse_args()

    # data_mode_list = ['native', 'auged']
    # model_mode_list = ['mymodel', 'tlearn']


    select_data = 'auged'
    select_model = 'mymodel'
    print("\nuse data:{} | model:{}".format(select_data, select_model))
    for i in range(3):
        print("\ndata no. {} -------------------------------".format(i))
        result_dict = main(data_mode=select_data,
                           model_mode=select_model,
                           no=i,
                           do_es=args.earlystopping)
        if i == 0:
            df_result = pd.DataFrame(result_dict.values(), index=result_dict.keys())
            """
                ['last_loss',
                 'last_acc',
                 'last_val_loss',
                 'last_val_acc',
                 'n_confuse',
                 'eval_loss',
                 'eval_acc',
                 'elapsed_time']
            """

        else:
            series = pd.Series(result_dict)
            df_result[i] = series
        print(df_result)

    csv_file = "./{}_{}_result.csv".format(select_data, select_model)
    df_result.to_csv(csv_file)

    print("\nexport {}  as CSV.".format(csv_file))

