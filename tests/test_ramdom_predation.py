import unittest
import sys

import numpy as np

 

from evolvepy.generator.basic_layers import RamdomPredation as RP

class TestSelection(unittest.TestCase):

	n_individuals = 100

	def test_selection(self):

		for i in range(10):
			population = [x for x in range(100)]
			fitness = np.sort(np.random.uniform(-100, 100, TestSelection.n_individuals))
			n_selected = 12
			rp = RP()
			selected, selected_fitness = rp(population, fitness)

			self.assertEqual(n_selected, len(selected))
			self.assertEqual(n_selected, len(selected_fitness))
			
			for i in range(n_selected):
				self.assertGreater(selected_fitness[i], fitness[-n_selected])
