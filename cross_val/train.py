
import os, sys
sys.path.append(os.pardir)
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
K.set_session(sess)


from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator, dataSplit

from sklearn.model_selection import StratifiedKFold




def main():

    cwd = os.getcwd()

    log_dir = os.path.join(cwd, "log")
    os.makedirs(log_dir, exist_ok=True)


    data_dir = os.path.join(cwd, "experiment_0")

    print("\ncreate data....")
    total_data, total_label = inputDataCreator(data_dir,
                                               224,
                                               normalize=True,
                                               #one_hot=True
    )
    print("\ntotal_data shape: ", total_data.shape)
    print("total_label shape: ", total_label.shape)

    input_size = total_data.shape[1]
    channel = total_data.shape[3]
    mh = ModelHandler(input_size, channel)

    skf = StratifiedKFold(n_splits=10)

    k = 0
    for traval_idx, test_idx in skf.split(total_data, total_label):
        print("\nK-Fold Cross-Validation k:{} ==========".format(k))

        print("\ntrain indices: \n", traval_idx)
        print("\ntest indices: \n", test_idx)

        test_data = total_data[test_idx]
        test_label = total_label[test_idx]

        print("-----*-----*-----")

        traval_data = total_data[traval_idx]
        traval_label = total_label[traval_idx]
        # print(traval_data.shape)
        # print(traval_label.shape)

        traval_label = np.identity(2)[traval_label.astype(np.int8)]
        test_label = np.identity(2)[test_label.astype(np.int8)]
        train_data, train_label, validation_data, validation_label, _, _ = dataSplit(traval_data,
                                                                                     traval_label,
                                                                                     train_rate=2/3,
                                                                                     validation_rate=1/3,
                                                                                     test_rate=0)
        print("train_data shape: ", train_data.shape)
        print("train_label shape: ", train_label.shape)
        print("validation_data shape: ", validation_data.shape)
        print("validation_label shape: ", validation_label.shape)
        print("test_data shape: ", test_data.shape)
        print("test_label shape: ", test_label.shape)

        print("*…*…*…*…*…*…*…*…*…*…*…*…*…*…*…*")

        model = mh.buildTlearnModel(base='mnv1')
        model.summary()

        history = model.fit(train_data,
                            train_label,
                            batch_size=10,
                            epochs=30,
                            validation_data=(validation_data, validation_label),
                            verbose=1)

        accs = history.history['accuracy']
        losses = history.history['loss']
        val_accs = history.history['val_accuracy']
        val_losses = history.history['val_loss']

        print("last val_acc: ", val_accs[len(val_accs)-1])

        print("\npredict sequence...")

        pred = model.predict(test_data,
                             #test_label,
                             batch_size=10,
                             verbose=1)

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


        print("\nevaluate sequence...")
        eval_res = model.evaluate(test_data,
                                  test_label,
                                  batch_size=10,
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

        print(save_dict)

        if k == 0:
            df_result = pd.DataFrame(save_dict.values(), index=save_dict.keys())
        else:
            series = pd.Series(save_dict)
            df_result[k] = series
        print(df_result)

        k+=1

    csv_file = "./result.csv"
    df_result.to_csv(csv_file)

    print("\nexport {}  as CSV.".format(csv_file))




if __name__ == '__main__':
    main()
