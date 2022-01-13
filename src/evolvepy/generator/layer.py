from __future__ import annotations
from abc import ABC, abstractmethod


from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from evolvepy.configurable import Configurable
from evolvepy.generator.context import Context

class Layer(Configurable):


	def __init__(self, name:str=None, dynamic_parameters:Dict[str, bool] = None, parameters:Dict[str, object] = None):
		super().__init__(parameters, dynamic_parameters, name=name)        

		self._next : List[Layer] = []
	   
		self._population = None
		self._fitness = None
		self._context = None

		self._prev_count : int = 0

	@property
	def next(self) -> List[Layer]:
		return self._next
	
	@next.setter
	def next(self, layer:Layer) -> None:
		if layer not in self._next:
			self._next.append(layer)

			layer._prev_count += 1

	@property
	def population(self) -> np.ndarray:
		return self._population
	
	@property
	def fitness(self) -> np.ndarray:
		return self._fitness

	@property
	def context(self) -> Context:
		return self._context

	def __call__(self, population:Union[ArrayLike, None], fitness:Union[ArrayLike, None]=None, context:Union[Context, None]=None) -> np.ndarray:          

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

	def __init__(self, name: str = None):
		super().__init__(name=name)

		self._received_count = 0

		self._population = None
		self._fitness = None

	def __call__(self, population: np.ndarray, fitness: np.ndarray, context:Union[Context, None]=None) -> Tuple[np.ndarray, np.ndarray]: # NOSONAR
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
	def __init__(self, name: str = None, dynamic_parameters: Dict[str, bool] = None, parameters: Dict[str, object] = None, chromosome_names: Union[str, List[str], None] = None):
		super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters)

		if isinstance(chromosome_names, str):
			self._chromosome_names = [chromosome_names]
		else:
			self._chromosome_names = chromosome_names

	
	def call(self, population:np.ndarray, fitness:np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:        
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
	
	def call_chromosomes(self, chromosomes:np.ndarray, fitness:np.ndarray, context:Context, name:Optional[str]) -> np.ndarray:
		return chromosomes