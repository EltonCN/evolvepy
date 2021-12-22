import unittest
import sys

import numpy as np
from numpy.testing import assert_equal, assert_raises
from numpy.testing._private.utils import assert_

from evolvepy.generator.context import Context

from .utils import assert_not_equal

 

from evolvepy.generator import Layer, Concatenate, FilterFirsts, Sort

class TestLayer(unittest.TestCase):

    def test_diverge(self):

        layer1 = Layer()
        layer2 = Layer()
        layer3 = Layer()

        layer1.next = layer2
        layer1.next = layer3

        pop = np.empty(10)
        fitness = np.empty(10)

        layer1(pop, fitness)

        assert_equal(pop, layer2.population)
        assert_equal(pop, layer3.population)

    def test_converge(self):
        layer1 = Layer()
        layer2 = Layer()
        layer3 = Layer()
        layer4 = Concatenate()

        layer1.next = layer2
        layer1.next = layer3
        layer2.next = layer4
        layer3.next = layer4

        pop = np.empty(10)
        fitness = np.empty(10)

        for _ in range(2):
            layer1(pop, fitness)

            pop_result = np.concatenate((pop, pop))
            fitness_result = np.concatenate((fitness, fitness))

            assert_equal(pop_result, layer4.population)
            assert_equal(pop.dtype, layer4.population.dtype)

            assert_equal(fitness_result, layer4.fitness)



    def test_filter(self):
        layer = FilterFirsts(5)

        pop = np.empty(10)
        fitness = np.empty(10)

        layer(pop, fitness)

        pop_result = pop[0:5]
        fitness_result = fitness[0:5]

        assert_equal(pop_result, layer.population)
        assert_equal(fitness_result, layer.fitness)

    def test_sort(self):
        layer = Sort()

        pop = np.arange(0, 1, 0.1)
        fitness = np.arange(0, 1, 0.1)

        layer(pop, fitness)

        pop_result = np.flip(pop)
        fitness_result = np.flip(fitness)

        assert_equal(fitness_result, layer.fitness)
        assert_equal(pop_result, layer.population)
        assert_equal(pop_result.shape, layer.population.shape)
        assert_equal(True, layer.context.sorted)

    
    def test_filter_sort(self):
        layer1 = Sort()
        layer2 = FilterFirsts(5)

        layer1.next = layer2

        pop = np.arange(0, 1, 0.1)
        fitness = np.arange(0, 1, 0.1)

        pop_result = np.flip(pop)[0:5]
        fitness_result = np.flip(fitness)[0:5]

        layer1(pop, fitness)

        assert_equal(pop_result, layer2.population)
        assert_equal(fitness_result, layer2.fitness)

    def test_block(self):
        layer = Sort()

        context = Context()

        context.block_all = True

        pop = np.arange(0, 1, 0.1)
        fitness = np.arange(0, 1, 0.1)

        layer(pop, fitness, context)

        assert_equal(layer.population, pop)
        assert_equal(layer.fitness, fitness)