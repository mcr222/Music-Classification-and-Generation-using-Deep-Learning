from __future__ import division
import numpy as np   
import scipy.io.wavfile as wav
import sounddevice as sd
import os
from subprocess import call
'''
The midi files in each folder are repetitions of the same song.
Removing repeated files: 

Songs don't contain sang lyrics.
Songs can be partitioned in 30 seconds with the same label.

From midi -> mono wav    timidity -oWM *.mid
From wav -> mp3 (lossless compression)     lame input_file.wav output_file.mp3
From mp3 -> wav            ffmpeg input_file.mp3 output_file.wav     pcm_s16le, 44100 Hz, mono, s16, 705 kb/s
MP3 compression works by reducing (or approximating) the accuracy of certain components of sound that are considered 
to be beyond the hearing capabilities of most humans. This method is commonly referred to as perceptual coding, 
or psychoacoustic modeling.

Reproduce raw wav     ffplay input_file.wav



'''
def readWavToNumpy(path):
    [rate, data] = wav.read(path)
    if(rate!=44100):
        print "NO correct sample rate!!!"
        
    print rate
    print len(data)
    print len(data)/rate
    print len(data)/rate/60
    #downsampling data by a factor of 10
    downsamplingfactor = 10
    datadown = data[0::downsamplingfactor]
    ratedown = rate//downsamplingfactor
    datadownlen = len(datadown)
    middle = datadownlen//2
    thirtysec = 30*ratedown
    data30sec = datadown[middle:middle+thirtysec]
    return data30sec


global_path = "ex_song/TRAXLZU12903D05F94/a.wav"
global_path = "ex_files/"

# print "Loading all labels to dictionary"
# genre_label_path = "../msd_tagtraum_cd2c.cls"
# genre_label_dictionary = {}
# with open(genre_label_path,'r') as genre_label_file:
#     i=0
#     for line in genre_label_file:
#         if i<10:
#             out = line.split("\t")
#             genre_label_dictionary[out[0]] = out[1].replace("\n","")
#         else:
#             break
#         i+=1
# 
# print genre_label_dictionary
# print genre_label_dictionary['TRAAAGF12903CEC202']
# print "Labels loaded"


full_path = "/home/mcr222/Documents/EIT/KTH/ScalableMachineLearning/MusicClassificationandGenerationusingDeepLearning/Music-Classification-and-Generation-using-Deep-Learning/"
fname= "a.mid"
output_filename = fname.replace(".mid","")
call("timidity -OwM " + full_path+ fname+ " "+ full_path+ output_filename +".wav "+ " ", shell=True)
#call("lame " + full_path+"/"+ output_filename +".wav" + " " +full_path+"/"+ output_filename+".mp3"+ " &> /dev/null", shell=True)

 
# for dirName, subdirList, fileList in os.walk(global_path):
#     print('Found directory: %s' % dirName)
#     first = True
#     for fname in fileList:
#         fullpath = dirName+"/"+fname
#         if(first):
#             track_ID = dirName.split("/")[-1]
#             print track_ID
#             first = False
#             output_filename = fname.replace(".mid","")
#             print fullpath
#             call("timidity -OwM "+ fullpath, shell=True)
#              
#             #label = genre_label_dictionary[track_ID]
#             
#             
#         os.remove(fullpath)
        

#This command only works on python command not in eclipse
# sd.play(data, rate)
# sd.play(datadown,ratedown)



