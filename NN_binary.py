from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Input, GlobalMaxPooling1D, Dropout
from keras.utils import normalize
from keras import optimizers

import numpy as np
import os
import keras
import random

# fix random seed for reproducibility
np.random.seed(7)


def createModel(filters,filters_end,window,window_end, learning_rate, dropout):
    '''Creates the NN model. As an input, the NN expects a normalized batch of 10 132300-dimensional
    For now, the following parameters are added to the model.
    @:param filters: number of filters in all the conv1d layers {50, 300}
    @:param learning_rate: learning rate of the model {0.0, 1.0}
    @:param dropout: Dropout value {0.0, 1.0}
        '''
    model = Sequential()
#     model.add(Embedding(1,
#                         100,
#                         input_length=132300))
#     model.add(Dropout(dropout))
    model.add(MaxPooling1D(5,input_shape = (132300,1)))
    model.add(Conv1D(filters, window, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters, window, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters_end, window_end, activation='relu'))
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    print(model.summary())

    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_test_split(all_files, test_size=0.3):
    lenfiles = len(all_files)
    lentest = int(lenfiles * test_size)
    lentrain = lenfiles - lentest

    test = random.sample(all_files, lentest)
    train = list(set(all_files) - set(test))
    # print type(test)
    # print type(train)
    random.shuffle(test)
    random.shuffle(train)
    return train, test


def main(filters = 128, filters_end=128,window=5,window_end=5, learning_rate = 0.01, dropout = 0.7, epochs = 10, batch_size = 100):
    '''
       Creates the NN model and train it with the batch data available in the input_training_folder file.
    '''

    # Global parameters
    # Input folder that contains all the subfolders A,B,C...containing the batch files.

    # input_folder = "output/"  #In hops
    #input_folder = "C:\\Users\\Diego\\Google Drive\\KTH\\Scalable Machine learning and deep learning\\project\\output"
    input_folder="/media/mcr222/First_Backup/smaller_output"

    # Parameters: For the NN model, initially three parameters are going to be considered as a parameter.
    # The number of filters, learning rate, dropout, epochs and batch size.
      # batch_size should be a multiple of 10.
    m = createModel(filters,filters_end,window,window_end, learning_rate, dropout)
    metrics = m.metrics_names
    print(metrics)

    # Split training and testing files
    all_files = []
    for folder in os.listdir(input_folder):
        print(folder)
        for file in os.listdir(input_folder + "/" + folder):
            if file.startswith("songs_"):
                all_files.append(input_folder + "/" + folder + "/" + file)

    training_files, test_files = train_test_split(all_files, test_size=0.3)
    #     print(all_files)
    #     print()
    #     print("Training files: " + str(training_files))
    #     print()
    #     print("Test files: " + str(test_files))
    #     print()
    print(len(all_files))
    print(len(training_files))
    print(len(test_files))

    train_results_loss = []
    train_results_acc = []
    test_results_loss = []
    test_results_acc = []

    # We train the model. For each file in the training folder, we extract the batch, normalize it and then call the function
    # train on batch from keras.
    concatenations = int(batch_size / 10)
    for epoch in range(epochs):
        print("------- Executing epoch " + str(epoch))
        random.shuffle(training_files)
        for i in range(0, len(training_files)):
            # training_batch will store the final training batch of size batch_size. listY will do the same for the labels
            listY = []
            training_batch = np.empty(shape=(0, 132300))
            count_labels = 0
            for j in range(0, concatenations):
                # Get input and label data from the same batch
                if i<len(training_files) and "songs_" in training_files[i]:
                    label = training_files[i].replace("songs", "labels")
                    print("training " + training_files[i])
                    data = np.load(str(training_files[i]))
                    normalizedData = normalize(data, axis=0, order=2)
                    print normalizedData.shape
                    training_batch = np.concatenate((training_batch, normalizedData), axis=0)

                    labels = np.load(str(label))

                    # print("New Labels")
                    # print(labels)
                    #TODO: change for np.reshape(a,(3,1))
                    for d in labels:
                        if(d == 10):
                            listY.append([1])
                            count_labels+=1
                        else:
                            listY.append([0])
                i += 1
            print("training_batch shape" + str(training_batch.shape))
            print("training batch labels size " + str(len(listY)))
            print count_labels
            #print(listY)

            # As the first value of the training_batch is an initial array containing zeros, we start training from the
            # first element of the array
            training_batch = np.expand_dims(training_batch, axis=2)
            
            x = m.train_on_batch(training_batch,listY)
            print("New batch: Train Loss " + str(x[0]) + " Train accuracy " + str(x[1]))
            train_results_loss.append(x[0])
            train_results_acc.append(x[1])

        # We test the model. For each file in the test folder, we extract the batch, normalize it and then call the function
        # test on batch from keras. Note that the test_on_batch function does not add to all batches, but it gives testing metrics
        # per batch. Thus, it is necessary to add all the results.
        # we only test every few batches (when running on hops we only need to test at the end)
        # TODO: only test on last epoch on hops
        if (epoch % 3 == 0 or epoch == epochs - 1):
            total_loss = 0
            total_accuracy = 0
            total_test_data = 0

            for file in test_files:
                if "songs_" in file:
                    label = file.replace("songs", "labels")
                    print("test " + file)
                    x_test = np.load(file)
                    x_test_normalized = normalize(x_test, axis=0, order=2)

                    labels_test = np.load(label)
                    y_test = []
                    for d in labels_test:
                        if(d == 10):
                            y_test.append([1])
                        else:
                            y_test.append([0])

                    x_test_normalized = np.expand_dims(x_test_normalized, axis=2)
                    x = m.test_on_batch(x_test_normalized, y_test)

                    print("New batch: Test Loss " + str(x[0]) + "Test accuracy " + str(x[1]))
                    total_test_data += 1
                    total_loss += x[0]
                    total_accuracy += x[1]

            test_results_loss.append(total_loss / total_test_data)
            test_results_acc.append(total_accuracy / total_test_data)
            print("[Total test files] " + str(total_test_data))
            print("[total_loss] " + str(total_loss) + " [total_loss] " + str(total_loss / total_test_data))
            print("[total_accuracy] " + str(total_accuracy) + " [total_accuracy] " + str(
                total_accuracy / total_test_data))

    print("Final training loss: ")
    print(train_results_loss)
    print("Final training accuracy: ")
    print(train_results_acc)
    print("Final test loss: ")
    print(test_results_loss)
    print("Final test accuracy: ")
    print(test_results_acc)
    
    print(str(total_accuracy/total_test_data))

    #return total_accuracy/total_test_data

#CAREFUL: with too many filters, big window and big batch_size one might run out of main memory (RAM)
main(filters = 20, filters_end=10,window=900,window_end=10, learning_rate = 0.014, dropout = 0, epochs = 10, batch_size = 200)

