import unittest
import sys

import numpy as np
from numpy.testing import assert_equal, assert_raises

from utils import assert_not_equal

sys.path.append("..\src")

from evolvepy.generator.layer import Layer

class TestLayer(unittest.TestCase):

    def test_diverge(self):

        layer1 = Layer()
        layer2 = Layer()
        layer3 = Layer()

        layer1.next = layer2
        layer1.next = layer3

        pop = np.ndarray(10)
        fitness = np.ndarray(10)

        layer1(pop, fitness)

        assert_equal(pop, layer2.population)
        assert_equal(pop, layer3.population)

    