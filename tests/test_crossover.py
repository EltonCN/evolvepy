import unittest
import sys

import numpy as np

import evolvepy.generator.crossover.crossover as crossover

class TestCrossover(unittest.TestCase):
    crossover_operators = [crossover.mean,  crossover.one_point, crossover.n_point]

    def test_crossover(self):

        chromosomes = np.random.rand(2, 100)

        for operator in TestCrossover.crossover_operators:
            new_ch = operator(chromosomes)

            self.assertEqual(type(new_ch), np.ndarray) #Correct type
            self.assertEqual(new_ch.shape, (100,)) #Correct shape

            np.testing.assert_equal(np.any(np.not_equal(new_ch, chromosomes[0])), True)
            np.testing.assert_equal(np.any(np.not_equal(new_ch, chromosomes[1])), True)