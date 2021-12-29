import unittest

import numpy as np
from numpy.testing import assert_equal
from numpy.testing._private.utils import assert_raises

from .utils import assert_not_equal

from evolvepy.callbacks import MemoryStoreLogger
from evolvepy.evaluator import FitnessCache, FunctionEvaluator, MultipleEvaluation
from evolvepy.configurable import Configurable
from evolvepy.generator import Descriptor, Generator

def sum1(individuals):
    return individuals[0]["chr0"].sum()

class TestLogger(unittest.TestCase):

    def test_evaluator_log(self):
        Configurable.reset_count()

        evaluator = FunctionEvaluator(sum1)
        dispatcher = MultipleEvaluation(evaluator)
        cache = FitnessCache(dispatcher)

        logger = MemoryStoreLogger(log_evaluator=True)

        descriptor = Descriptor()
        generator = Generator(descriptor=descriptor)

        logger.evaluator = cache
        logger.generator = generator

        logger.on_start()

        expected_log = {'FitnessCache2/n_generation': None, 'FitnessCache2/max_decimals': None, 'FitnessCache2/n_scores': 1, 'FitnessCache2/individual_per_call': 1, 'MultipleEvaluation1/agregator_name': 'mean', 'MultipleEvaluation1/n_scores': 1, 'MultipleEvaluation1/individual_per_call': 1, 'FunctionEvaluator0/evaluation_function_name': 'sum1', 'FunctionEvaluator0/n_scores': 1, 'FunctionEvaluator0/individual_per_call': 1}
    

        evaluator_log = logger._config_log['evaluator']


        for key in expected_log:
            assert_equal(key in evaluator_log, True)
            assert_equal(evaluator_log[key], expected_log[key])

