import unittest
import sys
import logging

import numpy as np
from numpy.testing._private.utils import assert_allclose
from numpy.typing import ArrayLike
from numpy.testing import assert_equal, assert_raises

from .utils import assert_not_equal

 

from evolvepy.evaluator.function_evaluator import FunctionEvaluator
from evolvepy.evaluator.dispatcher import MultipleEvaluation, EvaluationDispatcher
from evolvepy.evaluator.aggregator import FitnessAggregator
from evolvepy.evaluator.manager import EvaluationManager

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

def min_max(individuals:ArrayLike):
    fitness = np.empty(2, dtype=np.float64)
    fitness[0] = individuals[0]["chr0"].min()
    fitness[1] = individuals[0]["chr0"].max()

    return fitness

def get_population():
    dtype = np.dtype([("chr0", np.float32, 5)])
    population = np.empty(10, dtype)
    
    return population

class TestEvaluator(unittest.TestCase):

    def test_function_modes(self):
        population = get_population()
        fitness_reference = population["chr0"].sum(axis=1).reshape(10,1)

        fitness_result = []
        
        for mode in [FunctionEvaluator.PYTHON, FunctionEvaluator.JIT, FunctionEvaluator.NJIT, FunctionEvaluator.JIT_PARALLEL, FunctionEvaluator.NJIT_PARALLEL]:
            evaluator = FunctionEvaluator(sum1, mode=mode)
            fitness_result.append(evaluator(population))

        for fitness in fitness_result:
            assert_equal(fitness, fitness_reference)

    def test_function_individual_per_call(self):
        population = get_population()

        evaluator = FunctionEvaluator(sum2, mode=FunctionEvaluator.PYTHON, individual_per_call=2)

        fitness = evaluator(population)
        assert_equal(len(fitness), 10)
        assert_equal((fitness == 1).sum()+(fitness == -1).sum(), len(fitness))
        
    def test_dispatcher(self):
        population = get_population()
        fitness_reference = population["chr0"].sum(axis=1).reshape(10,1)

        evaluator = FunctionEvaluator(sum1)
        
        dispatcher = EvaluationDispatcher()
        fitness = dispatcher(population, evaluator)
        assert_equal(fitness, fitness_reference)

        dispatcher = MultipleEvaluation(n_evaluation=1)
        fitness = dispatcher(population, evaluator)
        assert_equal(fitness, fitness_reference)

    def test_aggregator(self):
        population = get_population()
        fitness_reference = population["chr0"].max(axis=1)

        evaluator = FunctionEvaluator(min_max, n_scores=2)
        aggre = FitnessAggregator(mode=FitnessAggregator.MAX)

        fitness = evaluator(population)
        fitness = aggre(fitness)

        assert_equal(fitness_reference, fitness)

    def test_manager(self):
        population = get_population()
        fitness_reference = population["chr0"].max(axis=1)

        evaluator = FunctionEvaluator(min_max, n_scores=2)
        dispatcher = MultipleEvaluation(n_evaluation=2)
        aggre = FitnessAggregator(mode=FitnessAggregator.MAX)

        manager = EvaluationManager(evaluator, dispatcher, aggre)

        fitness = manager(population)

        assert_equal(fitness_reference, fitness)

    
if __name__ == "__main__":
    #logging.basicConfig(level = logging.DEBUG, filename="error.log")
    TestEvaluator().test_function_modes()