import evolvepy

import numpy as np
from evolvepy.evaluator import FunctionEvaluator
from matplotlib import pyplot as plt
import ray

ray.shutdown()

ray.init()
def fitness_function(individuals):
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

evaluator = FunctionEvaluator.remote(fitness_function)

from evolvepy.generator import Descriptor

descriptor = Descriptor(chromosome_ranges=(-1000.0, 4000.0))

from evolvepy.generator import Generator, CombineLayer, Concatenate, Layer, FilterFirsts, ElitismLayer
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.generator.crossover import one_point
from evolvepy.generator.selection import tournament

# Defines the layers of the generator
first_layer = Layer() # Input layer 

# First path: Combine -> Mutation -> Filter
combine = CombineLayer(tournament, one_point)
mutation = NumericMutationLayer(sum_mutation, 1.0, 0.0, (-10.0, 10.0))
filter1 = FilterFirsts(95)
first_layer.next = combine
combine.next = mutation
mutation.next = filter1

# Second path: Sort -> Filter
elitism = ElitismLayer(5)
first_layer.next = elitism

# Combine both paths
concatenate = Concatenate()
filter1.next = concatenate
elitism.next = concatenate

# Creates the generator, specifying that an individual has only one chromosome, which can vary between -1000 and 4000 
generator = Generator(descriptor = descriptor, first_layer=first_layer, last_layer=concatenate)

from evolvepy import Evolver

evolver = Evolver(generator, evaluator, 100)

# Optmizes over 200 generations
hist, last_population = evolver.evolve(200) 

# Plot the results
plt.plot(hist.max(axis=1))
plt.plot(hist.mean(axis=1))

plt.legend(["Best", "Mean"])
plt.xlabel("Generation")
plt.ylabel("Fitness")

plt.ylim(800,1100)
plt.show()