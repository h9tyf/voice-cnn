import glob
import os

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')

model.predict()

test_file = "./test.wav"
wav_file, sr = librosa.load(test_file)

i = 0
length = 44100
step = int(length / 2)

dataSet = []
labels = []

parent_dir = './'
test_dir = 'test/'
file_name = '*.mp3'


cls = ['accrodion', 'guitar', 'piano', 'violin']
info = {}
count = 0
for label in cls:
    info[label] = count
    count += 1

print("start to read...")

t = 0
f = 0
for class_label in cls:
    file_path = parent_dir + test_dir + class_label + "/"
    test_files = glob.glob(os.path.join(file_path, file_name))
    print("processing ", class_label)
    print("path = ", file_path)
    print("file include:")
    print(test_files)
    for my_file in test_files:
        print("my_file = ", my_file)
        temp = np.load(my_file)
        for x in temp['arr_0']:
            label_predict = model.predict(x)
            print(label_predict)
            #if label_predict == info[class_label]:
            #    t += 1
            #else : f += 1



