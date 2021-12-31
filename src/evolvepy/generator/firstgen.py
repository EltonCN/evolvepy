from typing import Optional, Tuple, Union, List

import numpy as np
from numpy.typing import ArrayLike

from evolvepy.generator.descriptor import Descriptor
from evolvepy.generator.layer import ChromossomeOperator
from evolvepy.generator.context import Context

class FirstGenLayer(ChromossomeOperator):

	def __init__(self, descriptor:Descriptor, initialize_zeros:bool=False, name:str=None, chromossome_names: Union[str, List[str], None] = None):
		parameters = {"run":True, "initialize_zeros":initialize_zeros}
		dynamic_parameters = {"run":True}

		super().__init__(name=name, parameters=parameters, dynamic_parameters=dynamic_parameters, chromossome_names=chromossome_names)        

		self._descriptor = descriptor
		self._dtype = descriptor.dtype
		self._names = descriptor.chromossome_names
		self._chromossome_ranges = descriptor.chromossome_ranges

	def _generate_chromossome(self, population_size:int, name:str) -> np.ndarray:

		index = -1
		for i in range(self._descriptor._n_chromossome):
			if self._descriptor.chromossome_names[i] == name:
				index = i
				break
		if index == -1:
			raise RuntimeError("Chromossome name not in descriptor. Can't generate chromossome.")
		
		n_gene = self._descriptor._chromossome_sizes[index]
		name = self._names[index]
		dtype = self._dtype[name]
		shape = (population_size, n_gene)
		chromossome_range = self._chromossome_ranges[index]


		if dtype.base.char in np.typecodes["AllFloat"]:
			chromossome = np.random.rand(population_size, n_gene)
			chromossome *= chromossome_range[1] - chromossome_range[0]
			chromossome += chromossome_range[0]
		elif dtype.char in np.typecodes["AllInteger"]:
			chromossome = np.random.randint(chromossome_range[0], chromossome_range[1], shape)
		else:
			chromossome = np.random.choice([False, True], shape)

		return chromossome
	
	def __call__(self, population: Union[ArrayLike, None], fitness: Union[ArrayLike, None] = None, context: Union[Context, None] = None) -> np.ndarray:
		if population is None:
			if self.parameters["initialize_zeros"]:
				population = np.zeros(context.population_size, self._dtype)
			else:
				population = np.empty(context.population_size, self._dtype)
		return super().__call__(population, fitness=fitness, context=context)

	def call(self, population: np.ndarray, fitness: np.ndarray, context: Context) -> np.ndarray:
		if population is None and self.parameters["run"]:
			population = np.empty(context.population_size, dtype=self._dtype)
		return super().call(population, fitness, context)

	def call_chromossomes(self, chromossomes:np.ndarray, fitness:np.ndarray, context:Context, name:Optional[str]) -> np.ndarray:
		population_size = context.population_size

		#if self.parameters["run"]:
		#	self._parameters["run"] = False
		

		chromossomes =  self._generate_chromossome(population_size, name)

		return chromossomes