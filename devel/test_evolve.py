import unittest
import sys

import numpy as np
from numpy.testing import assert_equal, assert_raises

from evolvepy.generator.context import Context


from evolvepy.generator import Generator, Descriptor, descriptor, FirstGenLayer
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.generator.combine import CombineLayer
from evolvepy.generator.crossover import one_point
from evolvepy.generator.selection import tournament

layers = [  CombineLayer(tournament, one_point, 2),
					NumericMutationLayer(sum_mutation, 1.0, 0.5, (0.0, 1.0))]

descriptor = Descriptor(5)
gen = Generator(layers=layers, descriptor=descriptor)

dtype = np.dtype([("chr0", np.float32, 5)])

population = gen.generate(5)
first_pop = population.copy()

assert_equal(population.dtype, dtype)
assert_equal(population.shape, (5,))
assert_equal(population["chr0"].shape, (5, 5))

fitness = population["chr0"].sum(1)

gen.fitness = fitness

population = gen.generate(5)

assert_equal(population.dtype, dtype)
assert_equal(population.shape, (5,))
assert_equal(population["chr0"].shape, (5, 5))
#assert_not_equal(first_pop, population)

population = gen.generate(5)

assert_equal(population.dtype, dtype)
assert_equal(population.shape, (5,))
assert_equal(population["chr0"].shape, (5, 5))