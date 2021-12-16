import unittest
import sys
import logging

import numpy as np
from numpy.testing._private.utils import assert_allclose
from numpy.typing import ArrayLike
from numpy.testing import assert_equal, assert_raises

from utils import assert_not_equal

sys.path.append("..\src")

from evolvepy.evaluator.function_evaluator import FunctionEvaluator


def sum1(individuals:ArrayLike):
    return individuals[0]["chr0"].sum()

def sum2(individuals:ArrayLike):
    ind0_sum = individuals[0]["chr0"].sum()
    ind1_sum = individuals[1]["chr0"].sum()

    fitness = np.zeros(2, dtype=np.float64)
    if ind0_sum > ind1_sum:
        fitness[0] = 1
        fitness[1] = -1
    else:
        fitness[0] = -1
        fitness[1] = 1

    return fitness

class TestEvaluator(unittest.TestCase):

    def test_function_modes(self):
        dtype = np.dtype([("chr0", np.float32, 5)])
        population = np.empty(10, dtype)
        fitness_reference = population["chr0"].sum(axis=1)

        fitness_result = []
        
        for mode in [FunctionEvaluator.PYTHON, FunctionEvaluator.JIT, FunctionEvaluator.NJIT, FunctionEvaluator.JIT_PARALLEL, FunctionEvaluator.NJIT_PARALLEL]:
            evaluator = FunctionEvaluator(sum1, mode=mode)
            fitness_result.append(evaluator(population))

        for fitness in fitness_result:
            assert_equal(fitness, fitness_reference)

    def test_function_individual_per_call(self):
        dtype = np.dtype([("chr0", np.float32, 5)])
        population = np.empty(10, dtype)

        evaluator = FunctionEvaluator(sum2, mode=FunctionEvaluator.PYTHON, individual_per_call=2)

        fitness = evaluator(population)
        assert_equal(len(fitness), 10)
        assert_equal((fitness == 1).sum()+(fitness == -1).sum(), len(fitness))
        
    
if __name__ == "__main__":
    #logging.basicConfig(level = logging.DEBUG, filename="error.log")
    TestEvaluator().test_function_modes()