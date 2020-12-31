import glob
import os

import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def process_single(my_file, length, step):
    dataSet = []
    wav_file, sr = librosa.load(my_file)
    i = 0
    while i < len(wav_file) - length:
        temp_temp_temp = librosa.stft(wav_file[i: i + length])
        temp_temp = np.abs(temp_temp_temp)
        temp = librosa.amplitude_to_db(temp_temp, ref=np.max)
        temp = temp / -80.0
        dataSet.append(temp)
        i += step
    np.savez_compressed(my_file, np.array(dataSet, dtype=np.float16))


length = 44100
step = int(length / 2)

file_names = glob.glob('dataset/**/*.mp3', recursive=True)
for file in file_names:
    if not os.path.exists(file + ".npz"):
        print(file)
        process_single(file, length, step)
    else:
        print("Skipping", file)


