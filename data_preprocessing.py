from __future__ import division
import numpy as np   
import scipy.io.wavfile as wav
import sounddevice as sd
import os
from subprocess import call
import time

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
    print "Reading wav file: " + path
    i=0
    data30seclen=0
    data30secboth = []
    firstfragment = True
    secondfragment = True
    while len(data30secboth)!=2 and i<10:
        time.sleep(0.2)
        [rate, data] = wav.read(path)
        if(rate!=44100):
            print "NO correct sample rate!!!"
            
#         print rate
#         print len(data)
    #     print len(data)/rate
    #     print len(data)/rate/60
        #downsampling data by a factor of 10
        downsamplingfactor = 10
        datadown = data[0::downsamplingfactor]
        ratedown = rate//downsamplingfactor
        datadownlen = len(datadown)
        thirtysec = 30*ratedown
        fivesec = 5*ratedown
#         print thirtysec
        if fivesec+thirtysec < datadownlen and firstfragment:
            data30secboth.append(datadown[fivesec:fivesec+thirtysec])
            firstfragment = False
            
        if 2*fivesec+2*thirtysec < datadownlen and secondfragment:
            data30secboth.append(datadown[2*fivesec+thirtysec:2*fivesec+2*thirtysec])
            secondfragment = False

        i+=1
        #print "iiiiiii " + str(i)
    #print "data 30sec both " + str(len(data30secboth))
    return data30secboth


starttime = time.time()
print "Loading all labels to dictionary"
genre_label_path = "../msd_tagtraum_cd2c.cls"
genre_label_dictionary = {}
all_labels = {'Reggae': 0, 'Latin': 1, 'RnB': 2, 'Jazz': 3, 'Metal': 4, 'Pop': 5, 'Punk': 6, 'Country': 7, 'New Age': 8, 'Rap': 9, 'Rock': 10, 'World': 11, 'Blues': 12, 'Electronic': 13, 'Folk': 14}
with open(genre_label_path,'r') as genre_label_file:
    i=0
    for line in genre_label_file:
        if i>-1:
            out = line.split("\t")
            genre_label_dictionary[out[0]] = out[1].replace("\n","")
            #all_labels[out[1].replace("\n","")]=0
        else:
            break
        i+=1
        
print "DONE Loading all labels to dictionary"

#root_path = "/home/mcr222/Documents/EIT/KTH/ScalableMachineLearning/MusicClassificationandGenerationusingDeepLearning/Music-Classification-and-Generation-using-Deep-Learning/ex_files"
#root_path = "/media/mcr222/First_Backup/lmd_matched"
folder = "L"
root_path = "/home/mcr222/Documents/EIT/KTH/ScalableMachineLearning/MusicClassificationandGenerationusingDeepLearning/lmd_matched/"+folder
#output_folder = "/media/mcr222/First_Backup/output"
output_folder = "/home/mcr222/Documents/EIT/KTH/ScalableMachineLearning/MusicClassificationandGenerationusingDeepLearning/output/" + folder
songs = []
labels = []
batchno=0
all_labeled_songs = 0
all_iterated_songs = 0
all_iterated_files = 0
total_missed_wavs = 0
for dirName, subdirList, fileList in os.walk(root_path):
    print('Found directory: %s' % dirName)
    first = True
    for fname in fileList:
        full_path = dirName+"/"
        all_iterated_files += 1
        if(first):
            all_iterated_songs +=1
            track_ID = dirName.split("/")[-1]
            first = False
            if track_ID in genre_label_dictionary:
                label = all_labels[genre_label_dictionary[track_ID]]
                print track_ID
                print label
                output_filename = fname.replace(".mid","")
                call("timidity -OwM " + full_path+ fname+ " "+ full_path+ output_filename +".wav "+ " &> /dev/null", shell=True)
                time.sleep(3)
                try:
                    song = readWavToNumpy(full_path+output_filename +".wav")
                    
                    #print "songs added " + str(song)
                    songs.extend(song)
                    if(len(song)==1):
                        labels.append(label)
                    if(len(song)==2):
                        labels.extend([label,label])
                    
                    os.remove(full_path+output_filename +".wav")
                    all_labeled_songs +=1
                    print "Length of songs: " + str(len(songs))
                    if(len(songs)==10):
                        print "Saving batch number " + str(batchno)
                        np.save(output_folder+"/songs_10batch" +str(batchno)+".npy",songs)
                        np.save(output_folder+"/labels_10batch"+str(batchno)+".npy",labels)
                        songs = []
                        labels = []
                        batchno+=1
                    elif(len(songs)>10):
                        print "Saving batch number " + str(batchno)
                        np.save(output_folder+"/songs_10batch" +str(batchno)+".npy",songs[0:10])
                        np.save(output_folder+"/labels_10batch"+str(batchno)+".npy",labels[0:10])
                        songs = []
                        labels = []
                        songs.append(songs[-1])
                        labels.append(labels[-1])
                        batchno+=1
                        
                except:
                    print "Missed wav value!!!"
                    total_missed_wavs += 1
                    first = True
                    time.sleep(2)
                    try:
                        os.remove(full_path+output_filename +".wav")
                    except:
                        print "Could not remove file"
                
            else:
                print "No label for " + track_ID
              
        #os.remove(full_path+fname)
        
print "Total missed wavs " + str(total_missed_wavs)   
print "All iterated files "+str(all_iterated_files)
print "All iterated songs "+str(all_iterated_songs)
print "All labeled files " + str(all_labeled_songs)
print "Final batch number " + str(batchno)
print "Elapsed time " + str(time.time()-starttime)
#This is the last batch (not saved cause it does not have 10 songs)
np.save(output_folder+"/songs.npy",songs)
np.save(output_folder+"/labels.npy",labels)

#This command only works on python command not in eclipse
#songs = np.load('songs.npy')
#labels = np.load('labels.npy')
#ratedown=4410
# sd.play(data, rate)
# sd.play(songs[0],ratedown)



