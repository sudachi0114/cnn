
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

# define
img_size = 224
channel = 3
input_shape = (img_size, img_size, channel)

lim = 4


# ----------

model = VGG16(input_shape=input_shape,
              weights='imagenet',
              include_top=False)

model.summary()

print(model.layers)
print(len(model.layers))

# ----------

all_model = Sequential()
all_model.add(model)
all_model.add(Flatten())
all_model.add(Dense(256, activation='relu'))
all_model.add(Dropout(0.5))
all_model.add(Dense(1, activation='sigmoid'))

all_model.summary()

cwd = os.getcwd()
prj_root = os.path.dirname(cwd)

data_dir = os.path.join(prj_root, "dogs_vs_cats_smaller", "train")
cat_dir = os.path.join(data_dir, "cat")
print(cat_dir)

img_list = os.listdir(cat_dir)
img_list = sorted(img_list)

sample_location = os.path.join(cat_dir, img_list[1])
print(sample_location)

pil_obj = Image.open(sample_location)
pil_obj = pil_obj.resize((img_size, img_size))
sample = np.array(pil_obj)

print(sample.shape)

# plt.imshow(sample)
# plt.show()

sample = np.expand_dims(sample, axis=0)
print(sample.shape)

y = all_model.predict(sample)
print(y)

# -----------

"""
y_hw = model.predict(sample)
print(y_hw)
print(y_hw.shape)

# (1, 7, 7, 512) => (7, 7, 512)
y_hw = np.squeeze(y_hw, axis=0)
# (7, 7, 512) => (512, 7, 7)
y_hw = y_hw.transpose(2, 0, 1)
print(y_hw.shape)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(y_hw[i])
    plt.title(i)
    plt.axis(False)
plt.show()
"""

# -----------

layers = model.layers[:lim]
print(len(layers))

red_model = Sequential(layers)
red_model.summary()


y_hw = red_model.predict(sample)
#print(y_hw)
print(y_hw.shape)

# (1, 7, 7, 512) => (7, 7, 512)
y_hw = np.squeeze(y_hw, axis=0)
# (7, 7, 512) => (512, 7, 7)
y_hw = y_hw.transpose(2, 0, 1)
print(y_hw.shape)

for i in range(10):
    plt.subplot(2, 5, i+1)
    #plt.imshow(y_hw[i])
    plt.imshow(y_hw[i], cmap='gray')
    plt.title(i)
    plt.axis(False)
plt.show()

