import unittest
import sys

import numpy as np
from numpy.testing import assert_equal

from utils import assert_not_equal

sys.path.append("..\src")

from evolvepy.generator import CombineLayer
from evolvepy.generator.selection import tournament
from evolvepy.generator.crossover import one_point

class TestCombineLayer(unittest.TestCase):

    def test_single(self):
        population = np.random.rand(10, 10)
        population = np.round(population, 2)
        fitness = np.random.rand(10)
    
        layer = CombineLayer(tournament, one_point, 2, name="combine_test" )
        changed, _ = layer(population, fitness)

        assert_equal(changed.dtype, population.dtype)
        assert_not_equal(population, changed)


    def test_all(self):
        population = np.ones((10), dtype=[("chr0", np.float64, 10), ("chr1", np.float64, 2)])
        population[5:]["chr0"] *= 2.0
        population[5:]["chr1"] *= 2.0
        fitness = np.random.rand(10)

        layer = CombineLayer(tournament, one_point, 2, name="combine_test" )
        changed, _ = layer(population, fitness)

        assert_equal(changed.dtype, population.dtype)
        assert_not_equal(changed["chr0"], population["chr0"])
        assert_not_equal(changed["chr1"], population["chr1"])

    def test_one(self):
        population = np.ones((10), dtype=[("chr0", np.float64, 10), ("chr1", np.float64, 2)])
        population[5:]["chr0"] *= 2.0
        population[5:]["chr1"] *= 2.0
        fitness = np.random.rand(10)

        layer = CombineLayer(tournament, one_point, 2, name="combine_test", chromossome_names="chr0")
        changed, _ = layer(population, fitness)

        assert_equal(changed.dtype, population.dtype)
        assert_not_equal(changed["chr0"], population["chr0"])
        assert_equal(changed["chr1"], population["chr1"])