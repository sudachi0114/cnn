
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

from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold

from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator, dataSplit



def main(N, LEARN_PATH, MODE, BUILD_MODEL, EPOCHS=60, BATCH_SIZE=20, FINE_TUNE_AT=81):


    total_data, total_label = inputDataCreator(
        os.path.join(LEARN_PATH, "natural"),
        224,
        normalize=True)
        #one_hot=True

    print("\ntotal_data shape: ", total_data.shape)
    print("total_label shape: ", total_label.shape)

    if MODE == 'auged':
        auged_dir = os.path.join(LEARN_PATH, "auged")
        EPOCHS = EPOCHS//2 

        total_auged_data, total_auged_label = inputDataCreator(
            auged_dir,
            224,
            normalize=True,
            one_hot=True)
        print("\n  total auged_data : ", total_auged_data.shape)


    input_size = total_data.shape[1]
    channel = total_data.shape[3]

    mh = ModelHandler(input_size, channel)


    skf = StratifiedKFold(n_splits=5)

    k = 0
    for traval_idx, test_idx in skf.split(total_data, total_label):
        print("\nK-Fold Cross-Validation k:{} ==========".format(k))

        print("\ntrain indices: \n", traval_idx)
        print("\ntest indices: \n", test_idx)

        test_data = total_data[test_idx]
        test_label = total_label[test_idx]

        print(test_data)

        print("-----*-----*-----")

        traval_data = total_data[traval_idx]
        traval_label = total_label[traval_idx]
        # print(traval_data.shape)
        # print(traval_label.shape)

        traval_label = np.identity(2)[traval_label.astype(np.int8)]
        test_label = np.identity(2)[test_label.astype(np.int8)]

        train_data, train_label, validation_data, validation_label, _, _ = dataSplit(
            traval_data,
            traval_label,
            train_rate=3/4,
            validation_rate=1/4,
            test_rate=0)

        if MODE == 'auged':
            print("\nadd auged data to train_data...")

            auged_traval_data = total_auged_data[traval_idx]
            auged_traval_label = total_auged_label[traval_idx]

            auged_train_data, auged_train_label, _, _, _, _ = dataSplit(
                auged_traval_data,
                auged_traval_label,
                train_rate=3/4,
                validation_rate=1/4,
                test_rate=0)

            print("  append auged data: ", auged_train_data.shape)
            print("\n  concatnate auged data with native data...")
            train_data = np.vstack((train_data, auged_train_data))
            train_label = np.vstack((train_label, auged_train_label))
            print("    Done.")



        print("\ntrain data shape: ", train_data.shape)
        print("train label shape: ", train_label.shape)
        print("\nvalidation data shape: ", validation_data.shape)
        print("validation label shape: ", validation_label.shape)
        print("\ntest data shape: ", test_data.shape)
        print("test label shape: ", test_label.shape)


        es = EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           restore_best_weights=True,
                           mode='auto')

        print("set epochs: ", EPOCHS)


        if BUILD_MODEL == 'mymodel':
            model = mh.buildMyModel()

            # normal train ----------
            print("\ntraining sequence start .....")
            start = time.time()

            history = model.fit(
                train_data,
                train_label,
                BATCH_SIZE,
                epochs=EPOCHS,
                vlidation_data=(validation_data, validation_label),
                callbacks=[es],
                verbose=2)
            elapsed_time = time.time() - start

            
        elif BUILD_MODEL == 'tlearn':
            # あとで重みの解凍をできるように base_model を定義
            base_model = mh.buildMnv1Base()
            base_model.trainable=False

            model = mh.addChead(base_model)


            print("\ntraining sequence start .....")
            start = time.time()

            # 準備体操 -----
            print("\nwarm up sequence .....")
            model.summary()
            _history = model.fit(
                train_data,
                train_label,
                BATCH_SIZE,
                epochs=10,
                validation_data=(validation_data, validation_label),
                callbacks=[es],
                verbose=2)

            # fine tuning -----
            print("\nfine tuning.....")
            mh.setFineTune(base_model, model, FINE_TUNE_AT)
            model.summary()

            history = model.fit(
                train_data,
                train_label,
                BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(validation_data, validation_label),
                callbacks=[es],
                verbose=2)
            elapsed_time = time.time() - start



        # training end 
        accs = history.history['accuracy']
        losses = history.history['loss']
        val_accs = history.history['val_accuracy']
        val_losses = history.history['val_loss']


        log_dir = os.path.join(os.path.dirname(cwd), "flog")
        os.makedirs(log_dir, exist_ok=True)

        """
        child_log_dir = os.path.join(log_dir, "{}_{}_{}".format(MODE, BUILD_MODEL, no))
        os.makedirs(child_log_dir, exist_ok=True)

        # save model & weights
        model_file = os.path.join(child_log_dir, "{}_{}_{}_model.h5".format(MODE, BUILD_MODEL, no))
        model.save(model_file)

        # save history
        history_file = os.path.join(child_log_dir, "{}_{}_{}_history.pkl".format(MODE, BUILD_MODEL, no))
        with open(history_file, 'wb') as p:
            pickle.dump(history.history, p)

        print("\nexport logs in ", child_log_dir)
        """


        print("\npredict sequence...")
        pred = model.predict(test_data,
                             batch_size=10,
                             verbose=2)

        label_name_list = []
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

        eval_res = model.evaluate(test_data,
                                  test_label,
                                  batch_size=10,
                                  verbose=2)

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

        if k == 0:
            df_result = pd.DataFrame(save_dict.values(), index=save_dict.keys())
        else:
            series = pd.Series(save_dict)
            df_result[k] = series
        print(df_result)

        # undefine ----------
        # del total_data, total_label
        del traval_data, traval_label

        if MODE == 'auged':
            # del total_auged_data, total_auged_label
            del auged_traval_data, auged_traval_label
            del auged_train_data, auged_train_label

        del train_data, train_label
        del validation_data, validation_label
        del test_data, test_label
        
        del model
        del _history, history

        # clear session against OOM Error
        keras.backend.clear_session()
        gc.collect()

        k+=1

    csv_file = os.path.join( log_dir, "{}_{}_result.csv".format(MODE, BUILD_MODEL) )
    df_result.to_csv(csv_file)

    print("\nexport {}  as CSV.".format(csv_file))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DA 実験 複数例 学習プログラム")

    # parser.add_argument("--earlystopping", "-es", action="store_true",
    #                     help="学習時に EarlyStopping を ON にする.")

    args = parser.parse_args()

    data_mode_list = ['native', 'auged']
    model_mode_list = ['mymodel', 'tlearn']

    cwd = os.getcwd()
    sub_datasetsd = os.path.join(os.path.dirname(cwd),
                                 "subdatasets")

    N = 1
    for i in range(N):
        #    print("\ndata no. {} -------------------------------".format(i))
        learn_path = os.path.join(sub_datasetsd,
                                  "sample_{}".format(i))

        select_data = 'auged'
        select_model = 'tlearn'
        print("\nuse data:{} | model:{}".format(select_data, select_model))

        main(N=i,
             LEARN_PATH=learn_path,
             MODE=select_data,
             BUILD_MODEL=select_model,
             EPOCHS=60,
             BATCH_SIZE=20,
             FINE_TUNE_AT=81)

