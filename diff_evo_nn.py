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
def main(individual):
    '''
    Creates the NN model and train it with the batch data available in the input_training_folder file.
    '''


    #Parameters: For the NN model, initially two parameters are going to be considered as a parameter.
    # The number of filters and learning rate
    filters = individual[0]
    learning_rate = individual[1]
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
    pred = m.predict(X)
    print(pred)
    print(labels)
    count = 0

    for i, p in enumerate(pred):
        if p[0] == labels[0][i]:
            count += 1

    return count

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


# results = main()
# print(results)
#

'''
Differential evolution algorithm with parallelization option

This algorithm will create a full population to be evaluated, unlink typical differential evolution where each
individual get compared and selected sequentially. This allows the user to send a whole population of parameters
to a cluster and run computations in parallel, after which each individual gets evaluated with their respective
target or trial vector.
'''

import random

class DifferentialEvolution:
    _types = ['float', 'int', 'cat']
    _generation = 0
    _scores = []

    def __init__(self, objective_function, parbounds, types, direction = 'max', maxiter=10, popsize=10, mutationfactor=0.5, crossover=0.7):
        self.objective_function = objective_function
        self.parbounds = parbounds
        self.direction = direction
        self.types = types
        self.maxiter = maxiter
        self.n = popsize
        self.F = mutationfactor
        self.CR = crossover

        #self.m = -1 if maximize else 1

    # run differential evolution algorithms
    def solve(self):
        # initialise generation based on individual representation
        population, bounds = self._population_initialisation()
        print(population)
        for _ in range(self.maxiter):
            donor_population = self._mutation(population, bounds)
            trial_population = self._recombination(population, donor_population)
            population = self._selection(population, trial_population)

            new_gen_avg = sum(self._scores)/self.n

            if self.direction == 'max':
                new_gen_best = max(self._scores)
            else:
                new_gen_best = min(self._scores)
            new_gen_best_param = self._parse_back(population[self._scores.index(new_gen_best)])

            print("Generation: ", self._generation, " || ", "Average score: ", new_gen_avg,
                  ", best score: ", new_gen_best, "best param: ", new_gen_best_param)

        return population, self._scores

    # define bounds of each individual depending on type
    def _individual_representation(self):
        bounds = []

        for index, item in enumerate(self.types):
            b =()
            # if categorical then take bounds from 0 to number of items
            if item == self._types[2]:
                b = (0, len(self.parbounds[index]) - 1)
            # if float/int then take given bounds
            else:
                b = self.parbounds[index]
            bounds.append(b)
        return bounds

    # initialise population
    def _population_initialisation(self):
        population = []
        num_parameters = len(self.parbounds)
        for i in range(self.n):
            indiv = []
            bounds = self._individual_representation()

            for i in range(num_parameters):
                indiv.append(random.uniform(bounds[i][0], bounds[i][1]))
            indiv = self._ensure_bounds(indiv, bounds)
            population.append(indiv)
        return population, bounds

    # ensure that any mutated individual is within bounds
    def _ensure_bounds(self, indiv, bounds):
        indiv_correct = []

        for i in range(len(indiv)):
            par = indiv[i]

            # check if param is within bounds
            lowerbound = bounds[i][0]
            upperbound = bounds[i][1]
            if par < lowerbound:
                par = lowerbound
            elif par > upperbound:
                par = upperbound

            # check if param needs rounding
            if self.types[i] != 'float':
                par = round(par)
            indiv_correct.append(par)
        return indiv_correct

    # create donor population based on mutation of three vectors
    def _mutation(self, population, bounds):
        donor_population = []
        for i in range(self.n):

            indiv_indices = list(range(0, self.n))
            indiv_indices.remove(i)

            candidates = random.sample(indiv_indices, 3)
            x_1 = population[candidates[0]]
            x_2 = population[candidates[1]]
            x_3 = population[candidates[2]]

            # substracting the second from the third candidate
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
            donor_vec = [x_1_i + self.F*x_diff_i for x_1_i, x_diff_i in zip (x_1, x_diff)]
            donor_vec = self._ensure_bounds(donor_vec, bounds)
            donor_population.append(donor_vec)

        return donor_population

    # recombine donor vectors according to crossover probability
    def _recombination(self, population, donor_population):
        trial_population = []
        for k in range(self.n):
            target_vec = population[k]
            donor_vec = donor_population[k]
            trial_vec = []
            for p in range(len(self.parbounds)):
                crossover = random.random()

                # if random number is below set crossover probability do recombination
                if crossover <= self.CR:
                    trial_vec.append(donor_vec[p])
                else:
                    trial_vec.append(target_vec[p])
            trial_population.append(trial_vec)
        return trial_population

    # select the best individuals from each generation
    def _selection(self, population, donor_population):
        # Calculate trial vectors and target vectors and select next generation

        if self._generation == 0:
            for target_vec in population:
                parsed_target_vec = self._parse_back(target_vec)
                target_vec_score = self.objective_function(parsed_target_vec)
                self._scores.append(target_vec_score)

        for index, trial_vec in enumerate(donor_population):
            parsed_trial_vec = self._parse_back(trial_vec)
            trial_vec_score_i = self.objective_function(parsed_trial_vec)
            target_vec_score_i = self._scores[index]
            if self.direction == 'max':
                if trial_vec_score_i > target_vec_score_i:
                    self._scores[index] = trial_vec_score_i
                    population[index] = trial_vec
            else:
                if trial_vec_score_i < target_vec_score_i:
                    self._scores[index] = trial_vec_score_i
                    population[index] = trial_vec

        self._generation += 1

        return population

    # parse the converted values back to original
    def _parse_back(self, individual):
        original_representation = []
        for index, parameter in enumerate(individual):
            if self.types[index] == self._types[2]:
                original_representation.append(self.parbounds[parameter])
            else:
                original_representation.append(parameter)

        return original_representation

    # for parallelization purposes one can parse the population in dictionary format, adapt parameters accordingly
    def _parse_to_dict(self, population):
        population_dict = {'learning_rate' : [], 'dropout' : []}
        for indiv in population:
            population_dict['learning_rate'].append(indiv[0])
            population_dict['dropout'].append(indiv[1])

        return population_dict


if __name__ == "__main__":

    objective_fun = main

    diff_evo = DifferentialEvolution(objective_fun,[(100, 150),(0.001, 0.1)], ['int', 'float'], direction='max', maxiter=2,popsize=4)

    results = diff_evo.solve()

    print("Population: ", results[0])
    print("Scores: ", results[1])
