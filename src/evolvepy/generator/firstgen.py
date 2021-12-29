from typing import Tuple

import numpy as np

from evolvepy.generator.descriptor import Descriptor
from evolvepy.generator.layer import Layer
from evolvepy.generator.context import Context

class FirstGenLayer(Layer):

	def __init__(self, descriptor:Descriptor, name:str=None):
		parameters = {"run":True}
		dynamic_parameters = {"run":True}

		super().__init__(name=name, parameters=parameters, dynamic_parameters=dynamic_parameters)        

		self._descriptor = descriptor
		self._dtype = descriptor.dtype
		self._names = descriptor.chromossome_names
		self._chromossome_ranges = descriptor.chromossome_ranges

	def _generate_first(self, population_size:int) -> np.ndarray:

		population = np.empty(population_size, self._dtype)

		for i in range(self._descriptor._n_chromossome):
			n_gene = self._descriptor._chromossome_sizes[i]
			name = self._names[i]
			dtype = population[name].dtype
			shape = (population_size, n_gene)

			chromossome_range = self._chromossome_ranges[i]

			if dtype.char in np.typecodes["AllFloat"]:
				population[name] = np.random.rand(population_size, n_gene)
				population[name] *= chromossome_range[1] - chromossome_range[0]
				population[name] += chromossome_range[0]
			elif dtype.char in np.typecodes["AllInteger"]:
				population[name] = np.random.randint(chromossome_range[0], chromossome_range[1], shape)
			else:
				population[name] = np.random.choice([False, True], shape)

		return population
		
	def __call__(self, population:np.ndarray, fitness:np.ndarray=None, context:Context=None) -> Tuple[np.ndarray, np.ndarray]:
		population_size = context.population_size

		if self.parameters["run"]:
			self._parameters["run"] = False

			fitness = np.zeros(population_size, dtype=np.float32)
			population =  self._generate_first(population_size)
		if len(population) != population_size:
			size_difference = population_size - len(population)

			new_pop = self._generate_first(size_difference)
			new_fitness = np.zeros(size_difference)

			population = np.concatenate((population, new_pop))
			fitness = np.concatenate((fitness, new_fitness))

		self.send_next(population, fitness, context)