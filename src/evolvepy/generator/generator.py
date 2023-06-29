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
	Main class of the pipeline, it defines the layers oder, bifurcations and parameters
	'''

	def __init__(self, layers:Union[None, List[Layer]]=None, first_layer:Layer=None, last_layer:Layer=None, descriptor:Optional[Descriptor]=None):    
		'''
		Initialization for the Generator class with the desired layer order and individuasl description

		Args:
			layers (List[Layer]): List of layers used in the pipeline
			first_layer (Layer): First layer of the pipeline
			last_layer (Layer): Last layer of a pipeline
			descriptor (Descriptor): Object describing the individuals chromosome number, type and range
		'''

		self._connected = False

		if layers is None:
			layers = []
		elif first_layer is not None or last_layer is not None:
			raise ValueError("Generator 'layers' parameter must not be used together with 'first_layer' and 'last_layer'")
		else:
			layers = connect_layers(layers)

		if first_layer is not None and last_layer is not None:
			layers = connect_layers(first_layer, last_layer)

		elif last_layer is not None or first_layer is not None:
			raise ValueError("You must set Generator 'first_layer' with 'last_layer'")

		have_first_generator = False

		for layer in layers:
			if isinstance(layer, FirstGenLayer):
				have_first_generator = True

		if not have_first_generator:
			if descriptor is None:
				warnings.warn("You are creating a generator without FirstGenLayer and descriptor. Creating default descriptor.")
				
				descriptor = Descriptor()
			
			if len(layers) > 0 and isinstance(layers[-1], FirstGenLayer):
				raise RuntimeWarning("You are passing a descriptor, but also passing a FirstGenLayer. This can create unexpected behavior.")

			first_gen = FirstGenLayer(descriptor)

			if len(layers) > 0:
				layers[-1].next = first_gen

			layers.append(first_gen)
			self._descriptor = descriptor
		else:
			self._descriptor = layers[-1]._descriptor

		self._layers = layers

		self._fitness = None
		self._population = None

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
	
def connect_layers(layers: Union[None, List[Layer]] = None):
	if layers is None:
		return

	layer_ids = set()
	for i, layer in layers:
		if id(layer) in layer_ids:
			raise ValueError("Duplicate layer found in layers list, please check your layers list.")
		layer_ids.add(id(layer))
		if i < len(layers) - 1:
			layer.next = layers[i+1]
	
	return layers

def connect_layers(first_layer:Layer, last_layer:Layer):
	layers = []
	layers.append(first_layer)
	queue = deque()
	for layer in first_layer.next:
		queue.append(layer)
	
	while len(queue) != 0:
		layer:Layer = queue.pop()
		if layer != last_layer and layer not in layers:
			layers.append(layer)
			for next_layer in layer.next:
				queue.append(next_layer)
	layers.append(last_layer)

	layer_ids = set()
	for layer in layers:
		if id(layer) in layer_ids:
			raise ValueError("Duplicate layer found in layers list, please check your layers list.")
		layer_ids.add(id(layer))