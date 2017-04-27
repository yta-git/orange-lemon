import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.cross_validation import train_test_split

import glob
from PIL import Image
import matplotlib.pyplot as plt

from sys import exit


def plot_history(history):
 
    plt.plot(history.history['acc'], "o-", label="accuracy")
    plt.plot(history.history['val_acc'], "o-", label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    plt.plot(history.history['loss'], "o-", label="loss", )
    plt.plot(history.history['val_loss'], "o-", label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


image_train = np.empty((0, 3, 50, 50), dtype=np.uint8)
result_train = np.empty((0, 2), dtype=np.uint8)

for i in glob.glob("./orange/*"):
    img = Image.open(i).resize((50, 50))
    img_m = np.array(img).transpose(2, 0, 1)
    image_train = np.append(image_train, [img_m], axis=0)
    result_train = np.append(result_train, np.array([[1, 0]]), axis=0)

for i in glob.glob("./lemon/*"):
    img = Image.open(i).resize((50, 50))
    img_m = np.array(img).transpose(2, 0, 1)
    image_train = np.append(image_train, [img_m], axis=0)
    result_train = np.append(result_train, np.array([[0, 1]]), axis=0)

print(image_train.shape)

data_train, data_test, labels_train, labels_test = train_test_split(image_train, result_train, test_size=0.1,
                                                                    random_state=10)

model = Sequential()

model.add(Conv2D(16, (3, 3), border_mode='valid', input_shape=image_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(16, 3, 3, dim_ordering="th"))
model.add(Activation("relu"))
model.add(Dropout(0.25))


model.add(Conv2D(16, (3, 3), border_mode='valid', dim_ordering="th"))
model.add(Activation("relu"))
model.add(Conv2D(16, 3, 3, dim_ordering="th"))
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), border_mode='valid', dim_ordering="th"))
model.add(Activation("relu"))
model.add(Conv2D(16, 3, 3, dim_ordering="th"))
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), border_mode='valid', dim_ordering="th"))
model.add(Activation("relu"))
model.add(Conv2D(16, 3, 3, dim_ordering="th"))
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()
plot_model(model, to_file="model.png")
model.compile(loss='binary_crossentropy', optimizer="adadelta", metrics=['accuracy'])

history = model.fit(data_train, labels_train, batch_size=32, epochs=300, validation_data=(data_test, labels_test))

model_json_str = model.to_json()
open('model.json', 'w').write(model_json_str)
model.save_weights('weights.h5')
plot_history(history)
