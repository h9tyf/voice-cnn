import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('320_rs.wav')
plt.figure()

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

plt.subplot(1, 1, 1)
librosa.display.specshow(D, cmap='gray_r', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('对数频率功率谱')
# plt.title('2')
plt.show()

min_d = -80.0

D = D / min_d
print("1")
