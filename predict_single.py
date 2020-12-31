from tensorflow import keras
import glob
import numpy as np
from tqdm import tqdm
from collections import Counter

model = keras.models.load_model('model.pb')
cls = ['accrodion', 'guitar', 'piano', 'violin', 'voice']
array_length = 1025
array_width = 87


def get_cls(key):
    if key < len(cls):
        name = cls[key]
    else:
        name = str(key)
    return name

def ReLU(x):
    return x * (x > 0)


def predict_file(file):
    temp = np.load(file)
    predicts = []
    i = 0
    for x in tqdm(temp['arr_0']):
        ans = model.predict(x.reshape(1, array_length, array_width, 1))[0]
        ans = ReLU(ans)
        k = np.argmax(ans)
        confidence = ans[k] / np.sum(ans)
        # print("{:7f}s".format(i), "{:10}".format(get_cls(k)), "({:.1%} confidence)".format(confidence))
        predicts.append(k)
        i += 1
    ct = Counter(predicts)
    for key in ct:
        print("{} => {:.1%}".format(get_cls(key), ct[key] / len(predicts)))


files = glob.glob("dataset/test/**/*.npz")
for file in files:
    print("processing ", file)
    predict_file(file)
