import unittest
import sys

import numpy as np

 

import evolvepy.generator.mutation.numeric_mutation as mutation
import evolvepy.generator.mutation.binary_mutation as mutation_bin


class TestMutation(unittest.TestCase):
    operators = [mutation.sum_mutation, mutation.mul_mutation]
    operators_bin = [mutation_bin.bit_mutation]

    def test_numeric(self):

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

    def test_binary(self):
        chromossome = np.random.choice([0, 1], 1)

        new_ch = TestMutation.operators_bin[0](chromossome, 1.0, 0.0)

        print(chromossome, new_ch)