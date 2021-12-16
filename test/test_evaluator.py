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
    return individuals[0]["chr0"].sum() + individuals[1]["chr0"].sum()

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

        evaluator = FunctionEvaluator(sum2, mode=FunctionEvaluator.NJIT, individual_per_call=2)

        fitness = evaluator(population)
        fitness_reference  = [sum(population["chr0"].sum(1)[i:i+2]) for i in range(0, len(population["chr0"]), 2)]

        assert_allclose(fitness, fitness_reference)
    
if __name__ == "__main__":
    #logging.basicConfig(level = logging.DEBUG, filename="error.log")
    TestEvaluator().test_function_modes()