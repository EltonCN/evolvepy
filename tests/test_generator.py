import unittest
import sys

import numpy as np
from numpy.testing import assert_equal, assert_raises

from .utils import assert_not_equal

 

from evolvepy.generator import Generator
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.generator.combine import CombineLayer
from evolvepy.generator.crossover import one_point
from evolvepy.generator.selection import tournament

class TestGenerator(unittest.TestCase):

    def test_single(self):
        gen = Generator(10 , None, (-10.0, 50.0), np.float64, "single")

        population = gen.generate_first(10)

        dtype = np.dtype([("single", np.float64, 10)])

        assert_equal(population.dtype, dtype)
        assert_equal(population.shape, (10,))
        assert_equal(population["single"].shape, (10,10))

    def test_bool(self):
        gen = Generator(10, None, None, bool, "bool")

        population = gen.generate_first(5)

        dtype = np.dtype([("bool", bool, 10)])

        assert_equal(population.dtype, dtype)
        assert_equal(population.shape, (5,))
        assert_equal(population["bool"].shape, (5,10))

    def test_default(self):
        gen = Generator(5)

        population = gen.generate_first(1)

        dtype = np.dtype([("chr0", np.float32, 5)])

        assert_equal(population.dtype, dtype)
        assert_equal(population.shape, (1,))
        assert_equal(population["chr0"].shape, (1, 5))
    
    def test_evolve(self):
        layers = [  CombineLayer(tournament, one_point, 2),
                    NumericMutationLayer(sum_mutation, 1.0, 0.5, (0.0, 1.0))]

        gen = Generator(5, layers)
        dtype = np.dtype([("chr0", np.float32, 5)])

        population = gen.generate(5)
        first_pop = population.copy()

        assert_equal(population.dtype, dtype)
        assert_equal(population.shape, (5,))
        assert_equal(population["chr0"].shape, (5, 5))

        fitness = population["chr0"].sum(1)

        gen.fitness = fitness

        population = gen.generate()

        assert_equal(population.dtype, dtype)
        assert_equal(population.shape, (5,))
        assert_equal(population["chr0"].shape, (5, 5))
        assert_not_equal(first_pop, population)

    def test_error(self):
        gen = Generator(5)

        with assert_raises(RuntimeError):
            gen.generate()

    def test_get_all_parameters(self):
        layers = [  CombineLayer(tournament, one_point, 2),
                    NumericMutationLayer(sum_mutation, 1.0, 0.5, (0.0, 1.0))]

        gen = Generator(5, layers)

        dynamic_parameters = {}
        dynamic_parameters[layers[1].name+"/existence_rate"] = 1.0
        dynamic_parameters[layers[1].name+"/gene_rate"] = 0.5
        dynamic_parameters[layers[1].name+"/mutation_range_min"] = 0.0
        dynamic_parameters[layers[1].name+"/mutation_range_max"] = 1.0

        static_parameters = {}
        static_parameters[layers[0].name+"/selection_function_name"] = "tournament"
        static_parameters[layers[0].name+"/crossover_function_name"] = "one_point"
        static_parameters[layers[1].name+"/mutation_function_name"] = "sum_mutation"


        assert_equal(gen.get_all_dynamic_parameters(), dynamic_parameters)
        assert_equal(gen.get_all_static_parameters(), static_parameters)