import numpy as np

import keras
from keras.models import model_from_json
from PIL import Image
from keras import backend as K
import matplotlib.pyplot as plt

model = model_from_json(open("model.json").read())
model.load_weights("weights.h5")

model.compile(loss='binary_crossentropy', optimizer="adadelta", metrics=['accuracy'])

while True:
    try:
        test = np.empty((0, 3, 50, 50), dtype=np.uint8)
        img = Image.open(input())
        img.show()
        img = img.resize((50, 50))
        img_m = np.array(img).transpose(2, 0, 1)
        test = np.append(test, [img_m], axis=0)

        a = model.predict(test)

        o = round(a[0][0] * 100, 2)
        l = round(a[0][1] * 100, 2)

        print("This is probably", "orange." if o > l else "lemon.","({}%)".format(max(o, l)))
    except:
        pass