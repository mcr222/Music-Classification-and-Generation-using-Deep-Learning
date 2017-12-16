from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Input
from keras.utils import normalize
from keras import optimizers

import numpy as np
import os
# fix random seed for reproducibility
np.random.seed(7)


def createModel(filters, learning_rate):
    '''Creates the NN model. As an input, the NN expects a normalized batch of 10 132300-dimensional
    This is the Version 1 of the architecture, so it might change.
    '''
    #sequence_input = Input(shape=shape, dtype='float32')
    sequence_input = Input(shape=(10,132300), dtype='float32')
    #x = Embedding(output_dim=512, input_dim=10000, input_length=1000)(sequence_input)
    l_cov1 = Conv1D(filters, kernel_size=1, activation='relu', input_shape=(10,132300))(sequence_input)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(filters, kernel_size=1, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(1)(l_cov2)
    l_cov3 = Conv1D(filters, 1, activation='relu')(l_pool2)
    l_pool3 = MaxPooling1D(2)(l_cov3)  # global max pooling
    l_flat = Flatten()(l_pool3)
    l_dense = Dense(filters, activation='relu')(l_flat)
    preds = Dense(10, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)

    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    print("model fitting - simplified convolutional neural network")
    # m.summary()
    return model

'''
'''
def main():
    '''
    Creates the NN model and train it with the batch data available in the input_training_folder file.
    '''

    #Parameters: For the NN model, initially two parameters are going to be considered as a parameter.
    # The number of filters and learning rate
    filters = 128
    learning_rate = 0.01
    m = createModel(filters, learning_rate)

    #Input training Data
    input_training_folder = "./data/training_data"
    files = os.listdir(input_training_folder)
    for file in files:

        #Get input and label data from the same batch
        if file.startswith("songs"):
            batch = file.replace("songs", "")
            print("training " + file)
            data = np.load(input_training_folder + "\\" + str(file))
            X = normalizeData(data)

            labels = np.load(input_training_folder + "\\" + "labels" + str(batch))
            labels = np.expand_dims(labels, axis=0)

            m.train_on_batch(X, labels )


    #testing data
    # input_test_folder = ".\\data\\test_data"
    # X = np.load(input_test_folder + "\\songs_10batch1.npy")
    # xx = normalizeData(X)
    # Y =np.load(input_test_folder + "\\labels_10batch1.npy")
    # m.test_on_batch(xx,Y)


def normalizeData(data):
    normalizedData = normalize(data, axis=0, order=2)
    #print(data.shape)
    resp = np.expand_dims(normalizedData, axis=0)
    #print(X.shape)
    return resp


main()
