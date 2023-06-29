from typing import Dict, List, Tuple, Union, Optional
import warnings
from collections import deque

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from evolvepy.generator.context import Context
from evolvepy.generator.layer import Layer
from evolvepy.generator.firstgen import FirstGenLayer
from evolvepy.generator.descriptor import Descriptor

class Generator:
	'''
	Main class of the pipeline, it defines the layers order, bifurcations and parameters
	'''

	def __init__(self, layers: Union[None, List[Layer]] = None, first_layer: Layer = None, last_layer: Layer = None, descriptor: Optional[Descriptor] = None):
		'''
		Initialization for the Generator class with the desired layer order and individuals description

		Args:
			layers (List[Layer]): List of layers used in the pipeline
			first_layer (Layer): First layer of the pipeline
			last_layer (Layer): Last layer of a pipeline
			descriptor (Descriptor): Object describing the individuals chromosome number, type and range
		'''

		self._connected = False
		self._layers = layers or []
		self._fitness = None
		self._population = None

		if first_layer is not None or last_layer is not None:
			if layers is not None:
				raise ValueError("Generator 'layers' parameter must not be used together with 'first_layer' and 'last_layer'")
			self._initialize_with_first_and_last_layer(first_layer, last_layer)
		else:
			self._initialize_layers()

		self._initialize_first_gen_layer(descriptor)
		self._initial_population_size = len(self._population) if self._population is not None else None

		self.check_layers()

	def _initialize_with_first_and_last_layer(self, first_layer: Layer, last_layer: Layer):
		self._layers.append(first_layer)
		queue = deque(first_layer.next)

		while queue:
			layer = queue.pop()
			if layer != last_layer and layer not in self._layers:
				self._layers.append(layer)
				queue.extend(layer.next)

		self._layers.append(last_layer)

	def _initialize_layers(self):
		for i in range(len(self._layers) - 1):
			self._layers[i].next = self._layers[i + 1]

	def _initialize_first_gen_layer(self, descriptor: Optional[Descriptor]):
		have_first_generator = any(isinstance(layer, FirstGenLayer) for layer in self._layers)

		if not have_first_generator:
			if descriptor is None:
				warnings.warn("You are creating a generator without FirstGenLayer and descriptor. Creating default descriptor.")
				descriptor = Descriptor()

			if self._layers and isinstance(self._layers[-1], FirstGenLayer):
				raise RuntimeWarning("You are passing a descriptor, but also passing a FirstGenLayer. This can create unexpected behavior.")

			first_gen = FirstGenLayer(descriptor)

			if self._layers:
				self._layers[-1].next = first_gen

			self._layers.append(first_gen)
			self._descriptor = descriptor
		else:
			self._descriptor = self._layers[-1]._descriptor
	
	def set_parameter(self, layer_name:str, parameter_name:str, value:object) -> None:
		'''
		Defines the value of a specified parameter for a specified layer
		
		Args:
			layer_name (string): Name of the desired layer
			parameter_name (string): Name of th desired parameter
			value (object): Value to update the parameter
		
		'''
		for layer in self._layers:
			if layer.name == layer_name:
				layer.parameters = (parameter_name, value)
		self._check_population_size()

	def set_parameters(self, layer_name:str, parameters:Dict[str, object]) -> None:
		'''
		Defines the value of n specified parameters for a specified layer
		
		Args:
			layer_name (string): Name of the desired layer
			parameters (Dict[str, object]): Dict of parameters to update and their new values
		
		'''
		for layer in self._layers:
			if layer.name == layer_name:
				layer.parameters = parameters
		self._check_population_size()

	def get_parameter(self, layer_name:str, parameter_name:str=None) -> object:
		'''
		Recover the value of a specified parameter for a specified layer
		
		Args:
			layer_name (string): Name of the desired layer
			parameter_name (string): Name of th desired parameter
			
		Returns:
			parameter (object): Object containing the parameter values
		'''

		for layer in self._layers:
			if layer.name == layer_name:
				return layer.parameters[parameter_name]
		self._check_population_size()
	
	def get_parameters(self, layer_name:str) -> Dict[str, object]:
		'''
		Recover the value of all parameters for a specified layer
		
		Args:
			layer_name (string): Name of the desired layer
		
		Returns:
			parameters (Dict[str, object]): Dict of parameters and their values
		
		'''
		for layer in self._layers:
			if layer.name == layer_name:
				return layer.parameters
		self._check_population_size()

	def get_all_static_parameters(self) -> Dict[str, object]:
		'''
		Recover all static imutable parameters
		
		Returns:
			static_parameters (Dict[string, object]): Dictionay of pipeline's static parameters
		'''
		static_parameters = {}
		
		for layer in self._layers:
			name = layer.name
			layer_static_parameters = layer.static_parameters
			for key in layer_static_parameters:
				static_parameters[name+"/"+key] = layer_static_parameters[key]

		return static_parameters

	def get_all_dynamic_parameters(self) -> Dict[str, object]:
		'''
		Recover all static imutable parameters
		
		Returns:
			dynamic_parameters (Dict[string, object]): Dictionay of pipeline's dynamic parameters
		'''
		dynamic_parameters = {}

		for layer in self._layers:
			name = layer.name
			layer_dynamic_parameters = layer.dynamic_parameters
			for key in layer_dynamic_parameters:
				dynamic_parameters[name+"/"+key] = layer_dynamic_parameters[key]

		return dynamic_parameters

	def add(self, layer:Layer) -> None:
		'''
		Add layer to the pipeline
		
		Args:
			layer (Layer): Logical Layer to be added to the pipeline
		'''
		if len(self._layers) != 0:
			self._layers[-1].next = layer

		self._layers.append(layer)
	
	@property
	def fitness(self) -> np.ndarray:
		'''
		Return fitness atribute from population
		
		Returns:
			fitness (np.ndarray): Array of population fitness'''
		return self._fitness

	@fitness.setter
	def fitness(self, value:ArrayLike):
		'''
		Set fitness for each individual in the population
		
		Args:
			value (ArrayLike): Array of fitness values
		'''
		self._fitness = np.asarray(value)

	def generate(self, population_size:int) -> np.ndarray:
		'''
		Generate the pipeline from the given layers and atributes building the evolution model to pass to the evolver
		
		Args:
			population_size (int): Number of individuals in a population
			
		Returns:
			population (np.ndarray): generated population
		'''
		
		context = Context(population_size, self._descriptor.chromosome_names)

		if self._population is not None and len(self._population) > population_size:
			self._population = self._population[:population_size]

		self._layers[0](self._population, self._fitness, context)
		
		if len(self._layers[-1].population) != population_size:
			raise RuntimeError("The generator generated a population with wrong size. Expected "+str(population_size)+", got "+str(len(self._layers[-1].population)))

		self._population = self._layers[-1].population

		return self._population

	def check_layers(self):
		visited = set()
		connected_layers = set()
		layer_names = set()
		queue = deque([self._layers[0]])

		while queue:
			layer = queue.pop()
			connected_layers.add(layer)

			if layer.name in layer_names:
				raise ValueError("Duplicated layers found, please check your layers")
			layer_names.add(layer.name)

			for next_layer in layer.next:
				if next_layer in visited:
					raise ValueError("Loops detected between layers, please check your layers")
				visited.add(next_layer)
				if next_layer not in connected_layers:
					queue.append(next_layer)

		if len(connected_layers) != len(self._layers):
			raise ValueError("Unconnected layers found, please check your layers")

		if self._layers[-1] not in connected_layers:
			raise ValueError("First and last layers are not connected")

	def _check_population_size(self):
		if self._population is not None:
			current_population_size = len(self._population)
			if self._initial_population_size != current_population_size:
				warnings.warn(f"Population size changed from {self._initial_population_size} to {current_population_size}")