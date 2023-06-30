from __future__ import annotations
from abc import ABC, abstractmethod


from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from evolvepy.configurable import Configurable
from evolvepy.generator.context import Context

class Layer(Configurable):
	'''
	Base Layer class with essential properties and methods, can be used as base for especialized layers of as simple layer in the pipeline
	'''


	def __init__(self, name:str=None, dynamic_parameters:Dict[str, bool] = None, parameters:Dict[str, object] = None):
		'''
		Base initialization for a layer and it's dynamic and static parameters
		
		Args:
			name (string): Layer's name
			dynamic_parameters (Dict[string, bool]): Dictionary of mutable parameters
			static_parameters (Dict[string, object]): Dictionary of immutable parameters
		'''
		super().__init__(parameters, dynamic_parameters, name=name)        

		self._next : List[Layer] = []
	   
		self._population = None
		self._fitness = None
		self._context = None

		self._prev_count : int = 0

	@property
	def next(self) -> List[Layer]:
		'''
		Return the layer sucessor, so the generator and evolver only need to know the first and last layer of the pipeline.
		
		Returns:
			next(List[Layer]): List of immediate successors of the current layer, it is possible that the layers has n sucessors leading to n different pipes.
		'''
		return self._next
	
	@next.setter
	def next(self, layer:Layer) -> None:
		'''
		Set the layer sucessor, so the generator and evolver only need to know the first and last layer of the pipeline.
		
		Args:
			layer(Layer): Layer to succeed the current one.

		Returns:
			next(List[Layer]): List of immediate successor sucessor layers
		'''
		if layer not in self._next:
			self._next.append(layer)

			layer._prev_count += 1

	@property
	def population(self) -> np.ndarray:
		'''
		Returns the population array
		
		Retuns:
			population (np.ndarray): Population array
		'''
		return self._population
	
	@property
	def fitness(self) -> np.ndarray:
		'''
		Returns the fitness array
		
		Retuns:
			fitness (np.ndarray): Fitness array
		'''
		return self._fitness

	@property
	def context(self) -> Context:
		'''
		Returns the context object
		
		Retuns:
			context (Context): Context object for the next layers
		'''
		return self._context

	def population_size(self) -> int:
		'''
		Returns the number of previous layers
		
		Returns:
			prev_count (int): Number of previous layers
		'''
		return len(self._population) if self._population is not None else None

	def __call__(self, population:Union[ArrayLike, None], fitness:Union[ArrayLike, None]=None, context:Union[Context, None]=None) -> np.ndarray:
		'''
		Generic call to use the object as a funtion call, applying the layer oparetion
		
		Args:
			population (np.ndarray): Population array
			fitness (np.ndarray): Fitness array
			context (Context): Context object for the next layers

		Returns:
			population (np.ndarray): New population array
			fitness (np.ndarray): New fitness array
		'''         

		if not (population is None and fitness is None):
			population = np.asarray(population)

			if fitness is None:
				fitness = np.zeros(len(population), dtype=np.float32)
			fitness = np.asarray(fitness).flatten()

			if context is None:
				context = Context(len(population), population.dtype.names)

			if not context.block_all:
				population, fitness = self.call(population, fitness, context)

		
		self.send_next(population, fitness, context)
		

		self._context = context

		return population, fitness

	def send_next(self, population, fitness, context):
		'''
		Send context, population and fitness to the next layers
		
		Args:
			population (np.ndarray): Population array
			fitness (np.ndarray): Fitness array
			context (Context): Context object for the next layers
		'''
		self._population = population
		self._fitness = fitness

		for layer in self._next:
			next_context = context
			if len(self._next) != 1:
				next_context = next_context.copy()
				
			layer(population, fitness, next_context)

	def call(self, population:np.ndarray, fitness:np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
		return population, fitness


class Concatenate(Layer):
	'''
	Concatenation layer to join 2 or more layers
	'''

	def __init__(self, name: str = None):
		'''
		Initialization for thr Concatenate layer with the number of layers
		
		Args:
			name (string): Layer's name
		'''
		super().__init__(name=name)

		self._received_count = 0

		self._population = None
		self._fitness = None

	def __call__(self, population: np.ndarray, fitness: np.ndarray, context:Union[Context, None]=None) -> Tuple[np.ndarray, np.ndarray]: # NOSONAR
		'''
		Generic call to concatenate the layers using the object as a funtion call
		
		Args:
			population (np.ndarray): Population array
			fitness (np.ndarray): Fitness array
			context (Context): Context object for the next layers
		
		Returns:
			population (np.ndarray): New population array
			fitness (np.ndarray): New fitness array

		'''
		if not (population is None and fitness is None):
			population = np.asarray(population)

			if fitness is None:
				fitness = np.zeros(len(fitness), dtype=np.float32)
			fitness = np.asarray(fitness).flatten()

			if self._received_count == 0 or self._population is None:
				self._population = population
				self._fitness = fitness
			else:
				self._population = np.concatenate((self._population, population))
				self._fitness = np.concatenate((self._fitness, fitness))

		self._received_count += 1
		context.sorted = False

		if self._prev_count == self._received_count:
			self._received_count = 0
			self.send_next(self._population, self._fitness, context)

		self._context = context

		return population, fitness


class ChromosomeOperator(Layer):
	'''
	Base layer for chromossome operations such as numeric and boolean mutations
	'''

	def __init__(self, name: str = None, dynamic_parameters: Dict[str, bool] = None, parameters: Dict[str, object] = None, chromosome_names: Union[str, List[str], None] = None):
		'''
		Initialization for the chromossome operator layer with name and parameters
		
		Args:
			name (string): Layer's name
			dynamic_parameters (Dict[string, bool]): Dictionary of mutable parameters
			parameters (Dict[string, object]): Dictionary of immutable parameters
			chromosome_names (List[str]): Name of each chromosome
		'''

		super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters)

		if isinstance(chromosome_names, str):
			self._chromosome_names = [chromosome_names]
		else:
			self._chromosome_names = chromosome_names

	
	def call(self, population:np.ndarray, fitness:np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
		'''
		Generic call to use the object as a funtion call, applying the layer oparetion
		
		Args:
			population (np.ndarray): Population array
			fitness (np.ndarray): Fitness array
			context (Context): Context object for the next layers

		Returns:
			population (np.ndarray): New population array
			fitness (np.ndarray): New fitness array
		'''   

		result = population.copy()

		if self._chromosome_names is None: # Without specified name
			if len(population.dtype) == 0 and not context.blocked: # and only one chromosome
				result = self.call_chromosomes(population, fitness, context, None)
			else:
				for name in population.dtype.names: # and multiple chrmossomes
					if not context.blocked[name]:
						result[name] = self.call_chromosomes(population[name], fitness, context, name)
		else:
			for name in self._chromosome_names:
				if not context.blocked[name]:
						result[name] = self.call_chromosomes(population[name], fitness, context, name)


		return result, fitness
	
	#?????
	def call_chromosomes(self, chromosomes:np.ndarray, fitness:np.ndarray, context:Context, name:Optional[str]) -> np.ndarray:
		return chromosomes