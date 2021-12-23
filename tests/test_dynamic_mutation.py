import unittest
import sys

import numpy as np
from numpy.testing import assert_equal, assert_raises

from .utils import assert_not_equal

 

from evolvepy import Evolver
from evolvepy.evaluator import FunctionEvaluator
from evolvepy.generator import Generator
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.callbacks import DynamicMutation

def return_one(individuals):
    return 1

PRINT = False

def helper_print_stage(evolver: Evolver):

    if not PRINT:
        return

    dym_callback = None
    for callback in evolver._callbacks:
        if isinstance(callback, DynamicMutation):
            dym_callback = callback
            break
    
    mut_layer = None
    for layer in evolver._generator._layers:
        if isinstance(layer, NumericMutationLayer):
            mut_layer = layer
            break

    if dym_callback._stage == 0:
        print("NORMAL     ", end=" ")
    elif dym_callback._stage == 1:
        print("REFINEMENT ", end=" ")
    elif dym_callback._stage == 2:
        print("EXPLORATION", end=" ")

    print(dym_callback._step_count, end=" ")
    print(mut_layer.parameters["mutation_range"])


class TestDynamicMutation(unittest.TestCase):

    def test(self):
        layers = [NumericMutationLayer(sum_mutation, 1.0, 0.0, (0.0, 1.0))]
        generator = Generator(1, layers)

        evaluator = FunctionEvaluator(return_one)

        callbacks = [DynamicMutation([layers[0].name], patience=2, refinement_patience=3, exploration_patience=4, 
                                    refinement_steps=2, exploration_steps=2,  refinement_divider=5, exploration_multiplier=8)]

        evolver = Evolver(generator, evaluator, 1, callbacks)

        mutation_range = np.array([0.0, 1.0])

        for _ in range(2):
            evolver.evolve(1)
            assert_equal(callbacks[0]._stage, DynamicMutation.NORMAL)
            assert_equal(layers[0].parameters["mutation_range_min"], mutation_range[0])
            assert_equal(layers[0].parameters["mutation_range_max"], mutation_range[1])
            helper_print_stage(evolver)

        for _ in range(2):
            mutation_range /= 5
            for _ in range(3):
                evolver.evolve(1)
                assert_equal(callbacks[0]._stage, DynamicMutation.REFINEMENT)
                assert_equal(layers[0].parameters["mutation_range_min"], mutation_range[0])
                assert_equal(layers[0].parameters["mutation_range_max"], mutation_range[1])
                helper_print_stage(evolver)

        mutation_range = np.array([0.0, 1.0])

        for _ in range(2):
            mutation_range *= 8
            for _ in range(4):
                evolver.evolve(1)
                assert_equal(callbacks[0]._stage, DynamicMutation.EXPLORATION)
                assert_equal(layers[0].parameters["mutation_range_min"], mutation_range[0])
                assert_equal(layers[0].parameters["mutation_range_max"], mutation_range[1])
                helper_print_stage(evolver)


        


if __name__ == "__main__":
    TestDynamicMutation().test()