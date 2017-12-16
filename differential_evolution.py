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

    # for parallelization purposes one can parse the population in dictionary format
    def _parse_to_dict(self, population):
        population_dict = {'learning_rate' : [], 'dropout' : []}
        for indiv in population:
            population_dict['learning_rate'].append(indiv[0])
            population_dict['dropout'].append(indiv[1])

        return population_dict



if __name__ == "__main__":

    # example function
    def func2(x):
        # Beale's function, use bounds=[(-4.5, 4.5),(-4.5, 4.5)], f(3,0.5)=0.
        term1 = (1.500 - x[0] + x[0] * x[1]) ** 2
        term2 = (2.250 - x[0] + x[0] * x[1] ** 2) ** 2
        term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return term1 + term2 + term3

    objective_fun = func2

    diff_evo = DifferentialEvolution(objective_fun,[(-4.5, 4.5),(-4.5, 4.5)], ['float', 'float'], direction='min', maxiter=100,popsize=10)

    results = diff_evo.solve()

    print("Population: ", results[0])
    print("Scores: ", results[1])
