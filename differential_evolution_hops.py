'''
Differential evolution algorithm extended to allow for categorical and integer values for optimization of hyperparameter
space in Neural Networks, including an option for parallelization.

This algorithm will create a full population to be evaluated, unlike typical differential evolution where each
individual get compared and selected sequentially. This allows the user to send a whole population of parameters
to a cluster and run computations in parallel, after which each individual gets evaluated with their respective
target or trial vector.

User will have to define:
- Objective function to be optimized
- Bounds of each parameter (all possible values)
- The Types of each parameter, in order to be able to evaluate categorical, integer or floating values.
- Direction of the optimization, i.e. maximization or minimization
- Number of iterations, i.e. the amount of generations the algorithm will run
- The population size, rule of thumb is to take between 5-10 time the amount of parameters to optimize
- Mutation faction between [0, 2)
- Crossover between [0, 1], the higher the value the more mutated values will crossover
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

        parsed_back_population = []
        for indiv in population:
            parsed_back_population.append(self._parse_back(indiv))

        return parsed_back_population, self._scores

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
    def _selection(self, population, trial_population):
        # Calculate trial vectors and target vectors and select next generation

        if self._generation == 0:
            parsed_population = []
            for target_vec in population:
                parsed_target_vec = self._parse_back(target_vec)
                parsed_population.append(parsed_target_vec)

            parsed_population = self._parse_to_dict(parsed_population)
            self._scores = self.objective_function(parsed_population)

        parsed_trial_population = []
        for index, trial_vec in enumerate(trial_population):
            parsed_trial_vec = self._parse_back(trial_vec)
            parsed_trial_population.append(parsed_trial_vec)

        parsed_trial_population =  self._parse_to_dict(parsed_trial_population)
        trial_population_scores = self.objective_function(parsed_trial_population)

        for i in range(self.n):
            trial_vec_score_i = trial_population_scores[i]
            target_vec_score_i = self._scores[i]
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
                original_representation.append(self.parbounds[index][parameter])
            else:
                original_representation.append(parameter)

        return original_representation

    # for parallelization purposes one can parse the population from a list to a  dictionary format
    # User only has to add the parameters he wants to optimize to population_dict
    def _parse_to_dict(self, population):
        population_dict = {'init': [], 'optimizer': [], 'epochs': [], 'batches': [], 'activation_function1': [],
                           'activation_function2': [], 'activation_function3': []}
        for indiv in population:
            population_dict['init'].append(indiv[0])
            population_dict['optimizer'].append(indiv[1])
            population_dict['epochs'].append(indiv[2])
            population_dict['batches'].append(indiv[3])
            population_dict['activation_function1'].append(indiv[4])
            population_dict['activation_function2'].append(indiv[5])
            population_dict['activation_function3'].append(indiv[6])
        return population_dict


if __name__ == "__main__":

    # example function
    def func2(x):
        # Beale's function, use bounds=[(-4.5, 4.5),(-4.5, 4.5)], f(3,0.5)=0.
        term1 = (1.500 - x[0] + x[0] * x[1]) ** 2
        term2 = (2.250 - x[0] + x[0] * x[1] ** 2) ** 2
        term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return term1 + term2 + term3
    print("{0} ===== Start".format(datetime.now().isoformat()))
    start_time = timeit.default_timer()
    tensorboard_hdfs_logdir = tflauncher.launch(spark, wrapper)
    elapsed = timeit.default_timer() - start_time
    print "Elapsed time: " + str(elapsed)
    print("{0} ===== Stop".format(datetime.now().isoformat()))
    objective_fun = func2

    diff_evo = DifferentialEvolution(objective_fun,[(-4.5, 4.5),(-4.5, 4.5)], ['float', 'float'], direction='min', maxiter=100,popsize=10)

    results = diff_evo.solve()

    print("Population: ", results[0])
    print("Scores: ", results[1])
