from typing import Dict, List, Tuple, Union, Optional
import warnings

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from evolvepy.generator.context import Context
from evolvepy.generator.layer import Layer
from evolvepy.generator.firstgen import FirstGenLayer
from evolvepy.generator.descriptor import Descriptor

class Generator:

	def __init__(self, layers:Union[None, List[Layer]]=None, first_layer:Layer=None, last_layer:Layer=None, descriptor:Optional[Descriptor]=None):    
		self._connected = False

		if layers is None:
			layers = []
		elif first_layer is not None or last_layer is not None:
			raise ValueError("Generator 'layers' parameter must not be used together with 'first_layer' and 'last_layer'")
		else:
			for i in range(len(layers)-1):
				layers[i].next = layers[i+1]

		if first_layer is not None and last_layer is not None:
			layers.append(first_layer)
			layers.append(last_layer)
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
		for layer in self._layers:
			if layer.name == layer_name:
				layer.parameters = (parameter_name, value)

	def set_parameters(self, layer_name:str, parameters:Dict[str, object]) -> None:
		for layer in self._layers:
			if layer.name == layer_name:
				layer.parameters = parameters

	def get_parameter(self, layer_name:str, parameter_name:str=None) -> object:
		for layer in self._layers:
			if layer.name == layer_name:
				return layer.parameters[parameter_name]
	
	def get_parameters(self, layer_name:str) -> Dict[str, object]:
		for layer in self._layers:
			if layer.name == layer_name:
				return layer.parameters

	def get_all_static_parameters(self) -> Dict[str, object]:
		static_parameters = {}
		
		for layer in self._layers:
			name = layer.name
			layer_static_parameters = layer.static_parameters
			for key in layer_static_parameters:
				static_parameters[name+"/"+key] = layer_static_parameters[key]

		return static_parameters

	def get_all_dynamic_parameters(self) -> Dict[str, object]:
		dynamic_parameters = {}

		for layer in self._layers:
			name = layer.name
			layer_dynamic_parameters = layer.dynamic_parameters
			for key in layer_dynamic_parameters:
				dynamic_parameters[name+"/"+key] = layer_dynamic_parameters[key]

		return dynamic_parameters

	def add(self, layer:Layer) -> None:
		if len(self._layers) != 0:
			self._layers[-1].next = layer

		self._layers.append(layer)
	
	@property
	def fitness(self) -> np.ndarray:
		return self._fitness

	@fitness.setter
	def fitness(self, value:ArrayLike):
		self._fitness = np.asarray(value)

	def generate(self, population_size:int) -> np.ndarray:
		
		context = Context(population_size, self._descriptor.chromossome_names)

		if self._population is not None and len(self._population) > population_size:
			self._population = self._population[:population_size]

		self._layers[0](self._population, self._fitness, context)
		
		if len(self._layers[-1].population) != population_size:
			raise RuntimeError("The generator generated a population with wrong size. Expected "+str(population_size)+", got "+str(len(self._layers[-1].population)))

		self._population = self._layers[-1].population

		return self._population