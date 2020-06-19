
# 実際に使ったモデルの定義部分
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=INPUT_SHAPE))  # 入力層
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))   # 中間層1
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))  # 中間層2
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))  # 中間層3
model.add(MaxPooling2D((2,2)))
model.add(Flatten())                              # 全結合層接続のためFlattenする
model.add(Dense(512, activation='relu'))          # 全結合層
model.add(Dense(NUM_CLASS, activation='softmax')) # 出力層
