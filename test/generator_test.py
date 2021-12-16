import unittest
import sys

import numpy as np
from numpy.testing import assert_equal

from utils import assert_not_equal

sys.path.append("..\src")

from evolvepy.generator import Generator

class TestGenerator(unittest.TestCase):

    def test_single(self):
        gen = Generator(10, (-10.0, 50.0), np.float64, "single")

        population = gen.generate_first(10)

        dtype = np.dtype([("single", np.float64, 10)])

        assert_equal(population.dtype, dtype)
        assert_equal(population.shape, (10,))
        assert_equal(population["single"].shape, (10,10))

    def test_bool(self):
        gen = Generator(10, None, bool, "bool")

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