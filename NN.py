from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Input, GlobalMaxPooling1D,Dropout
from keras.utils import normalize
from keras import optimizers

import numpy as np
import os
import keras
import random

# fix random seed for reproducibility
np.random.seed(7)


def createModel(filters, learning_rate, dropout):
    '''Creates the NN model. As an input, the NN expects a normalized batch of 10 132300-dimensional
    For now, the following parameters are added to the model.
    @:param filters: number of filters in all the conv1d layers {50, 300}
    @:param learning_rate: learning rate of the model {0.0, 1.0}
    @:param dropout: Dropout value {0.0, 1.0}
        '''
    model = Sequential()
    model.add(Embedding(1,
                        100,
                        input_length=132300))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters, 5, activation='relu'))
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    model.add(Dense(15))

    print(model.summary())


    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_test_split(all_files, test_size=0.3):
    lenfiles = len(all_files)
    lentest = int(lenfiles*test_size)
    lentrain = lenfiles - lentest
    
    all_files = np.array(all_files)
    test = random.sample(all_files,lentest)
    train = set(all_files)-set(test)    
    
    return train,test

def main():
    '''
       Creates the NN model and train it with the batch data available in the input_training_folder file.
    '''

    #Global parameters
    #Input folder that contains all the subfolders A,B,C...containing the batch files.

    #input_folder = "small_train/small_train/"  #In hops
    input_folder= "C:\\Users\\Diego\\Google Drive\\KTH\\Scalable Machine learning and deep learning\\project\\output2"
    input_folder="/home/mcr222/Documents/EIT/KTH/ScalableMachineLearning/MusicClassificationandGenerationusingDeepLearning/output"

    #Parameters: For the NN model, initially three parameters are going to be considered as a parameter.
    # The number of filters and learning rate
    filters = 128
    learning_rate = 0.01
    dropout = 0.2
    m = createModel(filters, learning_rate, dropout)

    #Split training and testing files
    all_files = []
    for folder in os.listdir(input_folder):
        print(folder)
        for file in os.listdir(input_folder + "/" + folder):
            if file.startswith("songs_"):
                all_files.append(input_folder + "/" + folder + "/" + file)

    training_files,test_files = train_test_split(all_files, test_size=0.3)
    print(all_files)
    print()
    print("Training files: " + str(training_files))
    print()
    print("Test files: " + str(test_files))
    print()
    print(len(all_files))
    print(len(training_files))
    print(len(test_files))

    #We train the model. For each file in the training folder, we extract the batch, normalize it and then call the function
    #train on batch from keras.
     
    for file in training_files:
 
        #Get input and label data from the same batch
        if "songs_" in file:
            label = file.replace("songs", "labels")
            print("training " + file)
            data = np.load(str(file))
            normalizedData = normalize(data, axis=0, order=2)
 
            labels = np.load(str(label))
            listY = []
            for d in labels:
                listY.append([d])
            m.train_on_batch(normalizedData, keras.utils.to_categorical(listY, num_classes=15) )
 
    # We test the model. For each file in the test folder, we extract the batch, normalize it and then call the function
    # test on batch from keras. Note that the test_on_batch function does not add to all batches, but it gives testing metrics
    #per batch. Thus, it is necessary to add all the results.
     
    total_loss = 0
    total_accuracy = 0
    total_test_data = 0
 
    for file in test_files:
        if "songs_" in file :
            label = file.replace("songs", "labels")
            print("test " + file)
            x_test = np.load(file)
            x_test_normalized = normalize(x_test, axis=0, order=2)
 
            labels_test = np.load(label)
            y_test = []
            for d in labels_test:
                y_test.append([d])
 
            x = m.test_on_batch(x_test_normalized, keras.utils.to_categorical(y_test, num_classes=15))
 
            print("New batch: Test Loss " + str(x[0]) +  "Test accuracy " + str(x[1]))
            total_test_data +=1
            total_loss +=x[0]
            total_accuracy +=x[1]
 
    print("[Total test files] " + str(total_test_data))
    print("[total_loss] " + str(total_loss) + " [total_loss] " + str(total_loss/total_test_data))
    print("[total_accuracy] " + str(total_accuracy) + " [total_accuracy] " + str(total_accuracy/total_test_data))
 
    return total_accuracy/total_test_data

main()
