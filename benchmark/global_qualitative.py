import numpy as np
from matplotlib import pyplot as plt

import evolvepy as ep
from evolvepy.generator import Generator, CombineLayer, Layer, FilterFirsts, ElitismLayer, Concatenate
from evolvepy.generator.descriptor import Descriptor
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.generator.crossover import one_point
from evolvepy.generator.selection import tournament
from evolvepy.evaluator import FunctionEvaluator, MultipleEvaluation, FitnessCache, FitnessAggregator
from evolvepy.callbacks import DynamicMutation
from evolvepy import Evolver

def fitness_function(individuals):
    individual = individuals[0]["chr0"][0] 

    score = 0

    if individual < 500:
        score = individual
    elif individual < 1000:
        score = 1000 - individual
    elif individual < 2000:
        score = individual - 1000
    elif individual < 3000:
        score = 3000 - individual
    elif individual < 4000:
        score = 2.0*(individual - 3000)
    else:
        score = 2.0*(5000-individual)

    return np.array([score, score])


# Specify that an individual has only one chromosome, which can vary between -1000 and 4000 
descriptor = Descriptor(1, (-1000.0, 4000.0), [np.float32])

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

# Creates the generator
generator = Generator(descriptor = descriptor, first_layer=first_layer, last_layer=concatenate)



func_evaluator = FunctionEvaluator(fitness_function, n_scores=2)
aggregator_evaluator = FitnessAggregator(func_evaluator)
multiple_evaluator = MultipleEvaluation(aggregator_evaluator, 10, discard_max=True, discard_min=True)
cache_evaluator = FitnessCache(multiple_evaluator, 10, 4)

dyn_mut = DynamicMutation([mutation.name], # Target mutation layers
                patience=3, refinement_patience=3, exploration_patience=3, # How many generations without improvements wait before realizing it got stuck in each stage
                refinement_steps=2, exploration_steps=5, # How many refinement or exploration steps to do
                refinement_divider=2, exploration_multiplier=4) # By how much to divide/multiply

# Here we specify for Evolver to use the previously created generator and evaluator, in generations of 100 individuals.
evolver = Evolver(generator, cache_evaluator, 100, [dyn_mut])

hist, last_population = evolver.evolve(1000, verbose=True) 