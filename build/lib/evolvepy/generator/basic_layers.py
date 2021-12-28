from typing import Tuple
import numpy as np

from evolvepy.generator.context import Context

from .layer import Layer

class Sort(Layer):

	def __init__(self, name: str = None, ):
		super().__init__(name=name)

	def call(self, population: np.ndarray, fitness: np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
		indexs = np.argsort(fitness)
		indexs = np.flip(indexs)

		context.sorted = True

		return population[indexs], fitness[indexs]

class FilterFirsts(Layer):

	def __init__(self, n_to_pass:int=1, name: str = None):
		parameters = {"n_to_pass":n_to_pass}
		dynamic_parameters = {"n_to_pass":True}
		super().__init__(name=name, parameters=parameters, dynamic_parameters=dynamic_parameters)

	def call(self, population: np.ndarray, fitness: np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
		n_to_pass = self.parameters["n_to_pass"]
		
		return population[0:n_to_pass], fitness[0:n_to_pass]

class RamdomPredate(Layer):

	def __init__(self, n_to_predate:int=1, name: str = None):
		parameters = {"n_to_predate":n_to_predate}
		dynamic_parameters = {"n_to_predate":True}
		super().__init__(name=name, parameters=parameters, dynamic_parameters=dynamic_parameters)

	def call(self, population: np.ndarray, fitness: np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
		n_to_predate = self.parameters["n_to_predate"]
		population = population[0:-(n_to_predate)]
		fitness = fitness[0:-(n_to_predate)]
		indexes = np.random.choice(np.arange(population.size-1), size=n_to_predate)

		new_individuals = []
		new_fitness = []
		for i in range(n_to_predate):
			new_population[i] = population[indexes[i]]
			new_fitness[i] = fitness[indexes[i]]
		return new_population, new_fitness