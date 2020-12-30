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

dataSet = []
labels = []

second = 2
y, sr = librosa.load('320_rs.wav')
length = sr * second
step = int(length / 2)

print(1)

parent_dir = './dataSet_new/'
train_dir = 'train/'
test_dir = 'test/'

file_name1 = '*.wav'
file_name2 = '*.mp3'


batch_size = 32
array_length = 1025
array_width = 87

print("============================================================")

#cls = ['banjo', 'bass_clarinet', 'bassoon', 'cello', 'clarinet'
#       , 'contrabassoon', 'cor_anglais', 'double_bass', 'flute', 'french_horn'
#       , 'guitar', 'mandolin', 'oboe', 'people', 'percussion'
#       , 'piano', 'saxphone', 'trombone', 'trumpet', 'tuba'
#       , 'viola', 'violin']
#cls = ['flute', 'guitar', 'piano', 'trumpet', 'violin']
#cls = ['banjo', 'bassoon', 'cello', 'clarinet', 'guitar']
cls = ['accrodion', 'guitar', 'piano', 'violin']
info = {}
count = 0
for label in cls:
    info[label] = count
    count += 1

print("start to read...")

sum = 0

for class_label in cls:
    file_path = parent_dir + train_dir + class_label + "/"
    train_files = glob.glob(os.path.join(file_path, file_name1))
    print("processing ", class_label)
    print("path = ", file_path)
    #print("file include:")
    #print(train_files)

    for my_file in train_files:
        print("my_file = ", my_file)
        wav_file, sr = librosa.load(my_file)
        i = 0
        count = 0
        while i < len(wav_file) - length:
            count+=1
            temp_temp_temp = librosa.stft(wav_file[i: i + length])
            temp_temp = np.abs(temp_temp_temp)
            temp = librosa.amplitude_to_db(temp_temp, ref=np.max)
            temp = temp / -80.0
            dataSet.append(temp)
            labels.append(info[class_label])
            i += step
        print("in file ", my_file, "get => ", count)
        sum += count
    train_files = glob.glob(os.path.join(file_path, file_name2))
    #print("processing ", class_label)
    #print("path = ", file_path)
    #print("file include:")
    #print(train_files)
    for my_file in train_files:
        print("my_file = ", my_file)
        wav_file, sr = librosa.load(my_file)
        i = 0
        count = 0
        while i < len(wav_file) - length:
            count+=1
            temp = librosa.amplitude_to_db(np.abs(librosa.stft(wav_file[i: i + length])), ref=np.max)
            temp = temp / -80.0
            dataSet.append(temp)
            labels.append(info[class_label])
            i += step
        print("in file ", my_file, "get => ", count)
        sum += count

print("======================================================")
print("sum = ", sum)