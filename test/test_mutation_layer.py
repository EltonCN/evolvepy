import unittest
import sys

import numpy as np
from numpy.testing import assert_equal

from utils import assert_not_equal

sys.path.append("..\src")

from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation


class TestMutationLayer(unittest.TestCase):

    def test_single(self):
        population = np.zeros((10,10), np.float64)
    
        layer = NumericMutationLayer(sum_mutation, 1.0, 0.0, (0.0, 1.0), "mutation_test")

        changed = layer(population)

        assert_equal(changed.dtype, population.dtype)
        assert_not_equal(changed, population)

    def test_all(self):
        population = np.zeros((10), dtype=[("chr0", np.float64, 10), ("chr1", np.float64, 2)])

        layer = NumericMutationLayer(sum_mutation, 1.0, 0.5, (0.0, 1.0), "mutation_test")
        changed = layer(population)

        assert_equal(changed.dtype, population.dtype)
        assert_not_equal(changed["chr0"], population["chr0"])
        assert_not_equal(changed["chr1"], population["chr1"])

    def test_one(self):
        population = np.zeros((10), dtype=[("chr0", np.float64, 10), ("chr1", np.float64, 2)])

        layer = NumericMutationLayer(sum_mutation, 1.0, 0.5, (0.0, 1.0), "mutation_test", chromossome_names="chr0")
        changed = layer(population)

        assert_equal(changed.dtype, population.dtype)
        assert_not_equal(changed["chr0"], population["chr0"])
        assert_equal(changed["chr1"], population["chr1"])