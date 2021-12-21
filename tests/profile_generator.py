import sys
import cProfile

import numpy as np


 

from evolvepy.generator import Generator
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.generator.combine import CombineLayer
from evolvepy.generator.crossover import one_point
from evolvepy.generator.selection import tournament

layers = [  CombineLayer(tournament, one_point, 2),
            NumericMutationLayer(sum_mutation, 1.0, 0.5, (0.0, 1.0))]

gen = Generator(5, layers)
dtype = np.dtype([("chr0", np.float32, 5)])


population = gen.generate(5)
cProfile.run("gen.generate(5)" , sort=1)

first_pop = population.copy()

fitness = population["chr0"].sum(1)

gen.fitness = fitness

cProfile.run("gen.generate()", sort=1)
cProfile.run("gen.generate()", sort=1)
