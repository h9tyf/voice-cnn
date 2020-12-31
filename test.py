import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
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

array_length = 1025
array_width = 87

warnings.filterwarnings('ignore')

cls = ['accrodion', 'guitar', 'piano', 'violin', 'voice']

model = models.Sequential()
model.add(layers.Conv2D(16, (7, 5), activation='relu', input_shape=(array_length, array_width, 1)))
model.add(layers.MaxPooling2D((8, 2)))
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((6, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(cls)))

print(model.summary())

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

print("============================================================")

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
    for my_file in files:
        temp = np.load(my_file)
        for x in temp['arr_0']:
            dataSet.append(x)
            labels.append(info[class_label])

train_data, val_data, train_labels, val_labels = train_test_split(dataSet, labels, test_size = 0.3)


print("start to train...")

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(np.array(train_data).reshape(-1, array_length, array_width, 1), np.array(train_labels), epochs=5,
                    validation_data=(np.array(val_data).reshape(-1, array_length, array_width, 1), np.array(val_labels)))


model.save('model.pb')

