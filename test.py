import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

dataSet = []
labels = []

length = 44100
step = int(length / 2)

print(1)

parent_dir = './dataset/'
train_dir = 'train/'
test_dir = 'test/'

file_name = '*.npz'


batch_size = 32
array_length = 1025
array_width = 87

print("============================================================")

cls = ['accrodion', 'guitar', 'piano', 'violin']
info = {}
count = 0
for label in cls:
    info[label] = count
    count += 1

print("start to read...")

for class_label in cls:
    file_path = parent_dir + train_dir + class_label + "/"
    files = glob.glob(os.path.join(file_path, file_name))
    print("processing ", class_label)
    print("path = ", file_path)
    print("file include:", files)
    for my_file in files:
        print("my_file = ", my_file)
        temp = np.load(my_file)
        dataSet.append(temp)
        labels.append(info[class_label])

train_data, val_data, train_labels, val_labels = train_test_split(dataSet, labels, test_size = 0.3)


model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(array_length, array_width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())

print("start to train...")

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(np.array(train_data).reshape(-1, array_length, array_width, 1), np.array(train_labels), epochs=10,
                    validation_data=(np.array(val_data).reshape(-1, array_length, array_width, 1), np.array(val_labels)))


model.save('my_model.h5')

