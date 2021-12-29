from __future__ import annotations
from abc import ABC, abstractmethod


from typing import Dict, List, Tuple, Union

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
	def next(self) -> Layer:
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

	def __init__(self, name: str = None, dynamic_parameters: Dict[str, bool] = None, parameters: Dict[str, object] = None):
		super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters)

		self._received_count = 0

		self._population = None
		self._fitness = None

	def __call__(self, population: np.ndarray, fitness: np.ndarray, context:Union[Context, None]=None) -> Tuple[np.ndarray, np.ndarray]: # NOSONAR
		if not (population is None and fitness is None):
			population = np.asarray(population)

			if fitness is None:
				fitness = np.zeros(len(population), dtype=np.float32)
			fitness = np.asarray(fitness).flatten()

			context = Context(population.dtype.names)

			if self._received_count == 0 or self._population is None:
				self._population = population
				self._fitness = fitness
			else:
				self._population = np.concatenate((self._population, population))
				self._fitness = np.concatenate((self._fitness, fitness))

		self._received_count += 1

		if self._prev_count == self._received_count:
			self._received_count = 0
			self.send_next(self._population, self._fitness, context)

		self._context = context

		return population, fitness


class ChromossomeOperator(Layer):
	def __init__(self, name: str = None, dynamic_parameters: Dict[str, bool] = None, parameters: Dict[str, object] = None, chromossome_names: Union[str, List[str], None] = None):
		super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters)

		if isinstance(chromossome_names, str):
			self._chromossome_names = [chromossome_names]
		else:
			self._chromossome_names = chromossome_names

	
	def call(self, population:np.ndarray, fitness:np.ndarray, context:Context) -> np.ndarray:        
		result = population.copy()

		if self._chromossome_names is None: # Without specified name
			if len(population.dtype) == 0 and not context.blocked: # and only one chromossome
				result = self.call_chromossomes(population, fitness, context)
			else:
				for name in population.dtype.names: # and multiple chrmossomes
					if not context.blocked[name]:
						result[name] = self.call_chromossomes(population[name], fitness, context)
		else:
			for name in self._chromossome_names:
				if not context.blocked[name]:
						result[name] = self.call_chromossomes(population[name], fitness, context)


		return result, fitness
	
	def call_chromossomes(self, chromossomes:np.ndarray, fitness:np.ndarray, context:Context) -> np.ndarray:
		return chromossomes