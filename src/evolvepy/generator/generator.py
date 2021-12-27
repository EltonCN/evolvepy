from typing import Dict, List, Tuple, Union, Optional
import numpy as np

from numpy.typing import ArrayLike, DTypeLike

from evolvepy.generator.layer import Layer, Concatenate, ChromossomeOperator
from evolvepy.generator.firstgen import FirstGenLayer
from evolvepy.descriptor import Descriptor

class Generator:

	def __init__(self, layers:Union[None, List[Layer]]=None, first_layer:Layer=None, last_layer:Layer=None, descriptor:Optional[Descriptor]=None, n_individual:Optional[int]=None):    
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

		if descriptor is not None:
			if isinstance(layers[-1], FirstGenLayer):
				raise RuntimeWarning("You are passing a descriptor, but also passing a FirstGenLayer. This can create unexpected behavior.")

			if n_individual is None:
				raise ValueError("You must use Generator 'descriptor' and 'n_individual' parameters together")

			first_gen = FirstGenLayer(descriptor, n_individual)
			layers.append(first_gen)

		self._layers = layers

		self._fitness = None
		self._population = None
		self._n_individual = None
		self.n_individual = n_individual

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

	def generate_evolve(self) -> np.ndarray:
		self._layers[0](self._population, self._fitness)
		
		if len(self._layers[-1].population) != self._n_individual:
			raise RuntimeError("The generator generated a population with wrong size")

		return self._layers[-1].population

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

	def generate(self) -> np.ndarray:

		self._population = self.generate_evolve()

		return self._population