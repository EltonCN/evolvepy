from typing import Optional, Tuple, Union, List

import numpy as np
from numpy.typing import ArrayLike

from evolvepy.generator.descriptor import Descriptor
from evolvepy.generator.layer import ChromosomeOperator
from evolvepy.generator.context import Context

class FirstGenLayer(ChromosomeOperator):
	'''
	Generates the first population for the pipeline
	'''

	def __init__(self, descriptor:Descriptor, initialize_zeros:bool=False, name:str=None, chromosome_names: Union[str, List[str], None] = None, run:bool=True):
		'''
		Initialization fr the first population, it defines the base of the evolution

		Args:
			descriptor (class Descriptor): Descriptor object defining the individuals
			initialize_zeros (bool): Flag to indentify type of initialization, if True the population will init with zero value individuals
			name (string): Name of the layer
			chromosome_names (List[string]): Names of the chromossomes of each inidividual
			run (bool): Flag to define if the code will run or not, it is needed to make it run only on command

		'''
		parameters = {"run":run, "initialize_zeros":initialize_zeros}
		dynamic_parameters = {"run":True}

		super().__init__(name=name, parameters=parameters, dynamic_parameters=dynamic_parameters, chromosome_names=chromosome_names)        

		self._descriptor = descriptor
		self._dtype = descriptor.dtype
		self._names = descriptor.chromosome_names
		self._chromosome_ranges = descriptor.chromosome_ranges

	def _generate_chromosome(self, population_size:int, name:str) -> np.ndarray:
		'''
		Initialize each chromosome and calculate its fitness
		
		Args:
			population_size (int): Number of inidividuals in each generation
			name (string): Chromosome name

		Returns:
			chromosome (np.ndarray): new chromosome
		'''

		index = -1
		for i in range(self._descriptor._n_chromosome):
			if self._descriptor.chromosome_names[i] == name:
				index = i
				break
		if index == -1:
			raise RuntimeError("chromosome name not in descriptor. Can't generate chromosome.")
		
		n_gene = self._descriptor._chromosome_sizes[index]
		name = self._names[index]
		dtype = self._dtype[name]
		shape = (population_size, n_gene)
		chromosome_range = self._chromosome_ranges[index]


		if dtype.base.char in np.typecodes["AllFloat"]:
			chromosome = np.random.rand(population_size, n_gene)
			chromosome *= chromosome_range[1] - chromosome_range[0]
			chromosome += chromosome_range[0]
		elif dtype.base.char in np.typecodes["AllInteger"]:
			chromosome = np.random.randint(chromosome_range[0], chromosome_range[1], shape)
		else:
			chromosome = np.random.choice([0, 1], shape).astype(np.bool_)

		return chromosome
	
	def __call__(self, population: Union[ArrayLike, None], fitness: Union[ArrayLike, None] = None, context: Union[Context, None] = None) -> np.ndarray:
		if population is None:
			if self.parameters["initialize_zeros"]:
				population = np.zeros(context.population_size, self._dtype)
			else:
				population = np.empty(context.population_size, self._dtype)
		return super().__call__(population, fitness=fitness, context=context)

	def call(self, population: np.ndarray, fitness: np.ndarray, context: Context) -> Tuple[np.ndarray, np.ndarray]:

		'''
		Generic call to initialize the population nd its fitness
		
		Args:
			population (np.ndarray): Population array
			fitness (np.ndarray): Fitness array
			context(Context): COntext and restrictions from previous layers
			
		Returns:
			population (np.ndarray): New population array
			fitness (np.ndarray): New fitness array
		'''
		if population is None and self.parameters["run"]:
			population = np.empty(context.population_size, dtype=self._dtype)

		
		if self.parameters["run"]:
			self._parameters["run"] = False
			return super().call(population, fitness, context)
		else:
			return population, fitness

	def call_chromosomes(self, chromosomes:np.ndarray, fitness:np.ndarray, context:Context, name:Optional[str]) -> np.ndarray:
		'''
		Return chromosome values
		
		Args:
			chromosomes (np.ndarray): Array of chromosomes
			fitness (np.ndarray): Array of population fitness
			context (Context): Context from previous layers
			name (string) : Layer name

		Returns:
			chromosomes (np.ndarray): Array of chromosomes
		
		'''
		population_size = context.population_size

		chromosomes =  self._generate_chromosome(population_size, name)

		return chromosomes