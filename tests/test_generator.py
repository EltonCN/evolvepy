import unittest
import sys

import numpy as np
from numpy.testing import assert_equal, assert_raises

from evolvepy.generator.context import Context

from .utils import assert_not_equal


from evolvepy.generator import Generator, Descriptor, descriptor, FirstGenLayer
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.generator.combine import CombineLayer
from evolvepy.generator.crossover import one_point
from evolvepy.generator.selection import tournament

class TestGenerator(unittest.TestCase):

	def test_single(self):
		descriptor = Descriptor(10, (-10.0, 50.0), [np.float64], names = "single")
		gen = Generator(descriptor = descriptor)

		population = gen.generate(10)

		dtype = np.dtype([("single", np.float64, 10)])

		assert_equal(population.dtype, dtype)
		assert_equal(population.shape, (10,))
		assert_equal(population["single"].shape, (10,10))

	def test_bool(self):
		descriptor = Descriptor(chromossome_sizes= 10, types=bool, names = "bool")
		gen = Generator(descriptor = descriptor)

		population = gen.generate(5)

		dtype = np.dtype([("bool", bool, 10)])

		assert_equal(population.dtype, dtype)
		assert_equal(population.shape, (5,))
		assert_equal(population["bool"].shape, (5,10))

	def test_default(self):
		descriptor = Descriptor(1)
		gen = Generator(descriptor=descriptor)

		population = gen.generate(5)

		dtype = np.dtype([("chr0", np.float32, tuple([1]))])

		assert_equal(population.dtype, dtype)
		assert_equal(population.shape, (5,))
		assert_equal(population["chr0"].shape, (5, 1))
	
	def test_evolve(self):
		layers = [  CombineLayer(tournament, one_point, 2),
					NumericMutationLayer(sum_mutation, 1.0, 0.5, (0.0, 1.0))]

		descriptor = Descriptor(5)
		gen = Generator(layers=layers, descriptor=descriptor)
		
		dtype = np.dtype([("chr0", np.float32, 5)])

		population = gen.generate(5)
		first_pop = population.copy()

		assert_equal(population.dtype, dtype)
		assert_equal(population.shape, (5,))
		assert_equal(population["chr0"].shape, (5, 5))

		fitness = population["chr0"].sum(1)

		gen.fitness = fitness

		population = gen.generate(5)

		assert_equal(population.dtype, dtype)
		assert_equal(population.shape, (5,))
		assert_equal(population["chr0"].shape, (5, 5))
		assert_not_equal(first_pop, population)

	def test_get_all_parameters(self):
		layers = [  CombineLayer(tournament, one_point, 2),
					NumericMutationLayer(sum_mutation, 1.0, 0.5, (0.0, 1.0))]

		descriptor = Descriptor()
		gen = Generator(layers=layers, descriptor=descriptor)

		dynamic_parameters = {}
		dynamic_parameters[layers[1].name+"/existence_rate"] = 1.0
		dynamic_parameters[layers[1].name+"/gene_rate"] = 0.5
		dynamic_parameters[layers[1].name+"/mutation_range_min"] = 0.0
		dynamic_parameters[layers[1].name+"/mutation_range_max"] = 1.0
		dynamic_parameters[gen._layers[-1].name+"/run"] = True

		static_parameters = {}
		static_parameters[layers[0].name+"/selection_function_name"] = "tournament"
		static_parameters[layers[0].name+"/crossover_function_name"] = "one_point"
		static_parameters[layers[1].name+"/mutation_function_name"] = "sum_mutation"
		static_parameters[gen._layers[-1].name+"/initialize_zeros"] = False

		#print(dynamic_parameters)
		#print("--")
		#print(gen.get_all_dynamic_parameters())

		assert_equal(gen.get_all_dynamic_parameters(), dynamic_parameters)
		assert_equal(gen.get_all_static_parameters(), static_parameters)

	def test_descriptor(self):
		descriptor = Descriptor([1,2], [(0,1),(1,2)], [int, np.float32], ["a", "b"])
		dtype = np.dtype([("a", int, tuple([1])), ("b", np.float32, 2)])

		assert_equal(descriptor.dtype, dtype)
	
	def test_default_descriptor(self):
		descriptor = Descriptor(1)
		dtype = np.dtype([("chr0", np.float32, tuple([1]))])

		assert_equal(descriptor.dtype, dtype)
	
	def test_multiple_generator(self):
		descriptor = Descriptor((1,3), [(0,1), (3,5)], [np.float32, np.float32])
		context = Context(10, descriptor.chromossome_names)

		layer1 = FirstGenLayer(descriptor, chromossome_names=["chr0"], initialize_zeros=True)
		
		pop, _ = layer1(None, context=context)

		assert_equal(np.bitwise_and(pop["chr0"] <=1, pop["chr0"] >=0), True)
		assert_equal(pop["chr1"] == 0, True)

		layer2 = FirstGenLayer(descriptor, chromossome_names=["chr1"])

		layer1.next = layer2

		layer1(None, context=context)
		pop = layer2.population

		assert_equal(np.bitwise_and(pop["chr0"] <=1, pop["chr0"] >=0), True)
		assert_equal(np.bitwise_and(pop["chr1"] <=5, pop["chr1"] >=3), True)