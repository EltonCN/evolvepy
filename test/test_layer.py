import unittest
import sys

import numpy as np
from numpy.testing import assert_equal, assert_raises

from utils import assert_not_equal

sys.path.append("..\src")

from evolvepy.generator.layer import Layer, Concatenate

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

        layer1(pop, fitness)

        pop_result = np.concatenate((pop, pop))
        fitness_result = np.concatenate((fitness, fitness))

        assert_equal(pop_result, layer4.population)
        assert_equal(pop.dtype, layer4.population.dtype)

        assert_equal(fitness_result, layer4.fitness)
