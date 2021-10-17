import unittest
import sys

import numpy as np

sys.path.append("..\src")

import evolvepy.generator.mutation.numeric_mutation as mutation


class TestMutation(unittest.TestCase):
    operators = [mutation.sum_mutation, mutation.mul_mutation]

    def test_crossover(self):

        chromossome = np.random.rand(100)

        for operator in TestMutation.operators:
            for i in range(10):
                existence_rate = np.random.rand()
                gene_rate = np.random.rand()
                mutation_range = np.random.rand(2)

                mutation_range = np.sort(mutation_range)

                new_ch = operator(chromossome, existence_rate, gene_rate, mutation_range)

                self.assertEqual(type(new_ch), np.ndarray) #Correct type
                self.assertEqual(new_ch.shape, (100,)) #Correct shape
