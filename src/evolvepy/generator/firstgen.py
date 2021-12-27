import numpy as np

from evolvepy.generator.descriptor import Descriptor
from evolvepy.generator.layer import Layer, Concatenate, ChromossomeOperator

class FirstGenLayer(Layer):

	def __init__(self, descriptor:Descriptor, n_individual:int, name:str=None):
		parameters = {"run":True, "n_individual":n_individual}
		dynamic_parameters = {"run":True}

		super().__init__(parameters, dynamic_parameters, name=name)        

		self._descriptor = descriptor
		self._n_individual = n_individual
		self._dtype = descriptor.dtype
		self._names = descriptor.chromossome_names
		self._chromossome_ranges = descriptor.chromossome_ranges

	def _generate_first(self) -> np.ndarray:

		population = np.empty(self._n_individual, self._dtype)

		for i in range(self._n_chromossome):
			n_gene = self._chromossome_sizes[i]
			name = self._names[i]
			dtype = population[name].dtype
			shape = (n_individual, n_gene)

			chromossome_range = self._chromossome_ranges[i]

			if dtype.char in np.typecodes["AllFloat"]:
				population[name] = np.random.rand(n_individual, n_gene)
				population[name] *= chromossome_range[1] - chromossome_range[0]
				population[name] += chromossome_range[0]
			elif dtype.char in np.typecodes["AllInteger"]:
				population[name] = np.random.randint(chromossome_range[0], chromossome_range[1], shape)
			else:
				population[name] = np.random.choice([False, True], shape)

		return population
		
	def call(self, population:np.ndarray, fitness:np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
		if self.parameters["run"]:
			self._parameters["run"] = False

			fitness = np.zeros(self._n_individual, dtype=np.float32)

			return self._generate_first(self._n_individual), fitness
		else:
			return population, fitness