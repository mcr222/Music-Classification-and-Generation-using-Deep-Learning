from scipy import signal
import numpy as np


data = np.load("/home/mcr222/Documents/EIT/KTH/ScalableMachineLearning/MusicClassificationandGenerationusingDeepLearning/output/A/songs_10batch0.npy")

#1s of window non overlapping, we could parametrize this and also try to optimize it
windowlensecs=1
f,t,spectogram_data = signal.spectrogram(data[0],fs=4410,window=("tukey",windowlensecs*4410))

print len(f)
print len(t)
#print spectogram_data



# spectogram_data = np.zeros(10,)
# 
# for d in data: