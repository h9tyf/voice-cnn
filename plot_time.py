import librosa

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import wave

file = "./dataSet_new/train/violin/4.wav"

f = wave.open(file, 'rb')  # 调用wave模块中的open函数，打开语音文件。
params = f.getparams()  # 得到语音参数
nchannels, sampwidth, framerate, nframes = params[:4]  # nchannels:音频通道数，sampwidth:每个音频样本的字节数，framerate:采样率，nframes:音频采样点数
strData = f.readframes(nframes)  # 读取音频，字符串格式
wavaData = np.fromstring(strData, dtype=np.int16)  # 得到的数据是字符串，将字符串转为int型
wavaData = wavaData * 1.0/max(abs(wavaData))  # wave幅值归一化
wavaData = np.reshape(wavaData, [nframes, nchannels]).T  # .T 表示转置
f.close()

time = np.arange(0, nframes) * (1.0 / framerate)
time = np.reshape(time, [nframes, 1]).T
plt.plot(time[0, :nframes], wavaData[0, :nframes], c="b")
plt.xlabel("time(seconds)")
plt.ylabel("amplitude")
plt.title("violin")
plt.savefig('violin_1.jpg')  # 保存绘制的图形
plt.show()