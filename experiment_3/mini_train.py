
import os, sys
sys.path.append(os.pardir)

import argparse
parser = argparse.ArgumentParser(description="ミニマル学習プログラム")

parser.add_argument("--debug", type=int, choices=[1, 2, 3],
                    help="デバッグモードで実行 (Level を 1 ~ 3 の整数値で選択) ")

args = parser.parse_args()

debug_lv = args.debug
if debug_lv == None:
    debug_lv = 0


import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
K.set_session(sess)


from utils.model_handler import ModelHandler
from utils.img_utils import inputDataCreator


cwd = os.getcwd()

log_dir = os.path.join(cwd, "mini_log")
#os.makedirs(log_dir, exist_ok=True)

data_dir = os.path.join(cwd, "experiment_0")


print("\ncreate train data")
total_data, total_label = inputDataCreator(data_dir,
                                           224,
                                           normalize=True,
                                           one_hot=True)

class_num = len(total_label[0])
print(class_num)

amount = total_data.shape[0]    # 1000
each_class_amount = int(amount / class_num)

# devide data -----
total_cat_data = total_data[:each_class_amount]
total_dog_data = total_data[each_class_amount:]

# devide label -----
total_cat_label = total_label[:each_class_amount]
total_dog_label = total_label[each_class_amount:]

print("total_cat_data.shape: ", total_cat_data.shape)
print("total_cat_label.shape: ", total_cat_label.shape)


print("\ncreate test data")

train_rate=0.6
validation_rate=0.3
test_rate = 0.1

train_data, train_label = [], []
validation_data, validation_label = [], []
test_data, test_label = [], []

# amount == 1000 => e_c_amount == 500
train_size = int( each_class_amount*train_rate )  # 300
validation_size = int( each_class_amount*validation_rate )  # 150
test_size = int( each_class_amount*test_rate )  # 50

print("train_size: ", train_size)
print("validation_size: ", validation_size)
print("test_size: ", test_size)


# split cat data ----------
train_cat_data, rest_cat_data = np.split(total_cat_data, [train_size])
print("train_cat_data: ", train_cat_data.shape)
print("rest: ", rest_cat_data.shape)

validation_cat_data, test_cat_data = np.split(rest_cat_data, [validation_size])
print("validation_cat_data: ", validation_cat_data.shape)
print("test_cat_data: ", test_cat_data.shape)

# 初回は代入, 2回目以降は (v)stack
train_data = train_cat_data
validation_data = validation_cat_data
test_data = test_cat_data

# split cat label ----------
train_cat_label, rest_cat_label = np.split(total_cat_label, [train_size])
# print("train_cat_label: ", train_cat_label.shape)
# print("rest: ", rest_cat_label.shape)

validation_cat_label, test_cat_label = np.split(rest_cat_label, [validation_size])
# print("validation_cat_label: ", validation_cat_label.shape)
# print("test_cat_label: ", test_cat_label.shape)

# 初回は代入, 2回目以降は (v)stack
train_label = train_cat_label
validation_label = validation_cat_label
test_label = test_cat_label


# split dog data ----------
train_dog_data, rest_dog_data = np.split(total_dog_data, [train_size])
# print("train_dog_data: ", train_dog_data.shape)
# print("rest: ", rest_dog_data.shape)

validation_dog_data, test_dog_data = np.split(rest_dog_data, [validation_size])
# print("validation_dog_data: ", validation_dog_data.shape)
# print("test_dog_data: ", test_dog_data.shape)

train_data = np.vstack((train_data, train_dog_data))
validation_data = np.vstack((validation_data, validation_dog_data))
test_data = np.vstack((test_data, test_dog_data))

# split dog label ----------
train_dog_label, rest_dog_label = np.split(total_dog_label, [train_size])
# print("train_dog_label: ", train_dog_label.shape)
# print("rest: ", rest_dog_label.shape)

validation_dog_label, test_dog_label = np.split(rest_dog_label, [validation_size])
# print("validation_dog_label: ", validation_dog_label.shape)
# print("test_dog_label: ", test_dog_label.shape)

train_label = np.vstack((train_label, train_dog_label))
validation_label = np.vstack((validation_label, validation_dog_label))
test_label = np.vstack((test_label, test_dog_label))

"""
print(train_data.shape)
print(validation_data.shape)
print(test_data.shape)
print(test_label)
"""

# program test -----
print("\ntest sequence... ")

# train -----
cls0_cnt = 0
cls1_cnt = 0
for i in range(len(train_label)):
    if train_label[i][0] == 1:
        cls0_cnt += 1
    elif train_label[i][1] == 1:
        cls1_cnt += 1
assert cls0_cnt == cls1_cnt
print("  -> train cleared.")

# validation -----
cls0_cnt = 0
cls1_cnt = 0
for i in range(len(validation_label)):
    if validation_label[i][0] == 1:
        cls0_cnt += 1
    elif validation_label[i][1] == 1:
        cls1_cnt += 1
assert cls0_cnt == cls1_cnt
print("  -> validation cleared.")

# test -----
cls0_cnt = 0
cls1_cnt = 0
for i in range(len(test_label)):
    if test_label[i][0] == 1:
        cls0_cnt += 1
    elif test_label[i][1] == 1:
        cls1_cnt += 1
assert cls0_cnt == cls1_cnt
print("  -> test cleared.\n")


mh = ModelHandler(224, 3)

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
