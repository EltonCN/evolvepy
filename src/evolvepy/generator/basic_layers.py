from typing import List, Optional, Tuple, Union
import numpy as np

from evolvepy.generator.context import Context

from .layer import Layer

class Sort(Layer):
	'''
	Sorting class for the model pipeline, it sorts the population acording to fitness
	'''
	
	def __init__(self, name: str = None ):
		'''
		Generic initialization for the object, used only for the parent class

		Args:
			name (str): name of the layer
		'''
		super().__init__(name=name)

	def call(self, population: np.ndarray, fitness: np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
		'''
		Sort the population by fitness.

		Args:
			population (np.ndarray): 
			fitness (np.ndarray):
			context (Context):

		Returns:
			population (np.ArrayLike): Array of individuals.
			fitness (np.ArrayLike): Array of fitness per individual.
		'''
		indexs = np.argsort(fitness)
		indexs = np.flip(indexs)

		context.sorted = True

		return population[indexs], fitness[indexs]

class FilterFirsts(Layer):
	'''
	Filter to get the first n_to_pass individuals.
	'''
	
	def __init__(self, n_to_pass:int=1, name: str = None):
		'''
		Initialize the Filter from the top layer

		Args:
			n_to_pass (int): Number of best individuals selected.
			name (strig): Name of the layer.
		'''
		parameters = {"n_to_pass":n_to_pass}
		dynamic_parameters = {"n_to_pass":True}
		super().__init__(name=name, parameters=parameters, dynamic_parameters=dynamic_parameters)

	def call(self, population: np.ndarray, fitness: np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
		'''
		Generic call to funcion to use the class as funtion call

		Args:
			population (np.ArrayLike): array of individuals
			parameters (Dict(string, string)): parameters for the function
			dynamic_parameters (Dit(string, string)): dinamic parameters of the function

		Returns:
			n_to_pass (int): Number of best individuals selected
		'''
		n_to_pass = self.parameters["n_to_pass"]
		
		return population[0:n_to_pass], fitness[0:n_to_pass]

class Block(Layer):
	'''
	Block a determined chromosome for the next layers
	'''

	def __init__(self, chromosome_names:Optional[Union[List[str], str]]=None, run:bool=False, name:str = None):
		'''
		Initialization for the Block class with the blocked chromosomes.

		Args:
			chromosome_names (Union[List[str], str]): names of the chromosomes
			run (bool): define the flag to run the code only once
			name (string): name of the layer
		'''

		if isinstance(chromosome_names, str):
			chromosome_names = [chromosome_names]

		parameters={"run":run, "chromosome_names":chromosome_names}

		super().__init__(name=name, parameters=parameters)

	def call(self, population: np.ndarray, fitness: np.ndarray, context: Context) -> Tuple[np.ndarray, np.ndarray]:
		'''
		Generic call to apply the mutation function

		Args:
			population (np.ndarray): array of inidividuals
			fitness (np.ndarray): array of individuals fitness
			context (Context): context of the population from the previous layers

		Returns:
			population (np.ndarray): new array of inidividuals
			fitness (np.ndarray): new array of individuals fitness
		'''
		if self.parameters["run"]:
			if self.parameters["chromosome_names"] is not None:
				for name in self.parameters["chromosome_names"]:
					context.blocked[name] = True
			else:
				context.block_all = True

		return population, fitness

class RandomPredation(Layer):
	'''
	Replaces the last n_to_predate individuals for random ones
	'''
	
	def __init__(self, n_to_predate:int = 1, name: str = None):
		'''
		Initialization for the RandomPredation class with the number individuals to be replaced.

		Args:
			n_to_predate (int): number of individuals to be replaced.
			name (string): name of the layer
		'''
		parameters = {"n_to_predate":n_to_predate}
		dynamic_parameters = {"n_to_predate":True}
		super().__init__(name=name, parameters=parameters, dynamic_parameters=dynamic_parameters)

		self._sort = Sort()

	def call(self, population: np.ndarray, fitness: np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
		'''
		Generic call to apply the mutation function

		Args:
			population (np.ndarray): Array of inidividuals
			fitness (np.ndarray): Array of individuals fitness
			context (Context): Context of the population from the previous layers

		Returns:
			population (np.ndarray): New array of inidividuals
			fitness (np.ndarray): New array of individuals fitness
		'''
		if not context.sorted:
			population, fitness = self._sort(population, fitness, context)
			context = self._sort.context

		n_to_predate = self.parameters["n_to_predate"]
		indexes = np.random.choice(np.arange(population.size - n_to_predate), size=n_to_predate)

		new_population = population.copy()
		new_fitness = fitness.copy()

		index = population.shape[0]-1
		for i in range(n_to_predate):
			new_population[index] = population[indexes[i]]
			new_fitness[index] = fitness[indexes[i]]

			index -= 1

		return new_population, new_fitness


class ElitismLayer(Layer):
	'''
	Chooses the best n_to_pass individuals from the population
	'''
	def __init__(self, n_to_pass:int = 1, name: str = None):
		'''
		Generic initialization for the object, used only for the parent class

		Args:
			n_to_pass (int): Number of individuals to be selected. Defaults 1.
			name (str): Layer's name. Defaults None (will use class name).
		'''
		parameters = {"n_to_pass":n_to_pass}
		dynamic_parameters = {"n_to_pass":True}
		super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters)

		self._layer_sort = Sort()
		self._layer_filter = FilterFirsts(n_to_pass)

	def call(self, population: np.ndarray, fitness: np.ndarray, context: Context) -> Tuple[np.ndarray, np.ndarray]:
		'''
		Sort the population by fitness and then

		Args:
			population (np.ndarray): 
			fitness (np.ndarray):
			context (Context):

		Returns:
			population (np.ArrayLike): Array of individuals.
			fitness (np.ArrayLike): Array of fitness per individual.
		'''
		if context.sorted == False:
			population, fitness = self._layer_sort(population, fitness, context)
		
		population, fitness = self._layer_filter(population, fitness, context)
		
		return population, fitness