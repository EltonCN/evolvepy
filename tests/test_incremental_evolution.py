import unittest

import numpy as np
from numpy.testing import assert_equal, assert_raises


from .utils import assert_not_equal


from evolvepy.generator import Generator, Descriptor, FirstGenLayer, Block
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.callbacks import IncrementalEvolution

from evolvepy import Evolver
from evolvepy.evaluator import FunctionEvaluator

def return_one(individuals):
	return 1

class TestIncrementalEvolution(unittest.TestCase):

	def test(self):
		descriptor = Descriptor([3, 6], [(0.0, 5.0), (-10.0, 10.0)], types=[np.float32, np.float32])

		layer1 = Block("chr1")
		layer2 = NumericMutationLayer(sum_mutation, 1.0, 0.0, (0.0, 100.0))
		layer3 = FirstGenLayer(descriptor, initialize_zeros=True, chromossome_names="chr0")
		layer4 = FirstGenLayer(descriptor, chromossome_names="chr1", run=False)

		layers = [layer1, layer2, layer3, layer4]
		for i in range(len(layers)-1):
			layers[i].next = layers[i+1]

		gen = Generator(first_layer=layer1, last_layer=layer4, descriptor=descriptor)
		evaluator = FunctionEvaluator(return_one)
		inc_evol = IncrementalEvolution(2, block_layer=layer1, first_gen_layer=layer4)

		evolver = Evolver(gen, evaluator, 10, [inc_evol])

		
		evolver.evolve(1)
		
		pop = layer4.population
		assert_equal(np.bitwise_and(pop["chr0"] >= 0.0, pop["chr0"] <= 5.0), True)
		assert_equal(pop["chr1"] == 0.0, True)

		evolver.evolve(1)

		pop2 = layer4.population
		assert_equal(pop2["chr1"] == 0.0, True)
		assert_equal((pop["chr0"] != pop2["chr0"]).sum(axis=1), 1)

		evolver.evolve(1)

		pop3 = layer4.population
		assert_equal(np.bitwise_and(pop3["chr1"] >= -10.0, pop3["chr1"] <= 10.0), True)
		assert_equal((pop2["chr0"] != pop3["chr0"]).sum(axis=1), 1)
		assert_equal((pop2["chr1"] != pop3["chr1"]).sum(axis=1), 6)

		evolver.evolve(1)

		pop4 = layer4.population
		assert_equal((pop3["chr0"] != pop4["chr0"]).sum(axis=1), 1)
		assert_equal((pop3["chr1"] != pop4["chr1"]).sum(axis=1), 1)