from typing import Optional, Union, Tuple, List

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

class Descriptor:
	'''
	Object for the caracteristics of the individuals in population
	'''

	def __init__(self, chromosome_sizes:Optional[ArrayLike]=1, chromosome_ranges:Union[None, List[Union[None, Tuple]], Tuple]=None, types:Union[list, DTypeLike]=[np.float32], names:Union[list, str, None]=None):
		'''
		Defines the initial population caracteristics

		Args:
			chromosome_sizes (int): Number of chromosomes per individual
			chromosome_ranges (Tuple[float, float]): Range of possible values for each chromosome
			types (List[DTypeLikes]): Data type of chromosome
			names (List[string]): Names of the chromossomes

		'''
		chromosome_sizes = np.asarray(chromosome_sizes)

		if chromosome_sizes.shape == ():
			chromosome_sizes = np.array([chromosome_sizes])

		n_chromosome = len(chromosome_sizes)

		if names is None:
			names = []
		elif isinstance(names, str):
			names = [names]
		

		if chromosome_ranges is None:
			chromosome_ranges = [None] * n_chromosome
		elif isinstance(chromosome_ranges, tuple):
			chromosome_ranges = [chromosome_ranges]

		if not isinstance(types, list):
			types = [types]

		self._chromosome_sizes = chromosome_sizes
		self._n_chromosome = n_chromosome
		self._chromosome_ranges = chromosome_ranges
		
		self._create_dtype_names_ranges(names, types)
	
	def _create_dtype_names_ranges(self, names, types):
		'''
		Defines the maximum ranges based on the given data types
		
		Args:
			names (List[string]): Names for the chromosomes
			types (List[DtypeLike]): Data types of the chromosomes
		'''
		self._names = []

		dtype = []
		for i in range(self._n_chromosome):
			name = "chr"+str(i)
			if len(names)-1 >= i:
				name = names[i]
			self._names.append(name)

			size = np.atleast_1d(self._chromosome_sizes[i])
			size = tuple(size)

			dtype.append((name, types[i], size))

			if self._chromosome_ranges[i] is None:
				if np.dtype(types[i]).char in np.typecodes["AllFloat"]:
					self._chromosome_ranges[i] = (0.0, 1.0)
				elif np.dtype(types[i]).char in np.typecodes["AllInteger"]:
					self._chromosome_ranges[i] = (0, 10)
				else:
					self._chromosome_ranges[i] = (0, 1)

		self._dtype = np.dtype(dtype)

	@property
	def dtype(self):
		'''
		Returns:
			Data type of the individuals
		'''
		return self._dtype
	
	@property
	def chromosome_names(self):
		'''
		Returns:
			Chromosome names of the individuals
		'''
		return self._names

	@property
	def chromosome_ranges(self):
		'''
		Returns:
			Chromosome ranges of the individuals
		'''
		return self._chromosome_ranges