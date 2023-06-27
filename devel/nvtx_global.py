import numpy as np
from matplotlib import pyplot as plt

import evolvepy as ep
from evolvepy.generator import Generator, CombineLayer
from evolvepy.generator.descriptor import Descriptor
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.generator.crossover import one_point
from evolvepy.generator.selection import tournament
from evolvepy.evaluator import *
from evolvepy import Evolver



# Defines the layers of the generator
layers = [CombineLayer(tournament, one_point), NumericMutationLayer(sum_mutation, 1.0, 0.0, (-10.0, 10.0))]

# Specify that an individual has only one chromosome, which can vary between -1000 and 4000 
descriptor = Descriptor(1, (-1000.0, 4000.0), [np.float32])

# Creates the generator
generator = Generator(layers=layers, descriptor=descriptor)

def fitness_function(individuals):
    # Gets the first gene of chromosome "chr0" (default name), in the first and only individual
    individual = individuals[0]["chr0"][0] 

    score = 0

    if individual < 500:
        score = individual
    elif individual < 1000:
        score = 1000 - individual
    elif individual < 2000:
        score = individual - 1000
    else:
        score = 3000 - individual

    return score


func_evaluator = FunctionEvaluator(fitness_function)
multiple_evaluator = MultipleEvaluation(func_evaluator, 10, discard_max=True, discard_min=True)
cache_evaluator = FitnessCache(multiple_evaluator, 10, 4)

# Here we specify for Evolver to use the previously created generator and evaluator, in generations of 100 individuals.
evolver = Evolver(generator, cache_evaluator, 100)

hist, last_population = evolver.evolve(1000, verbose=True) 