from typing import List, Tuple, Union
import numpy as np
from numpy.lib.arraysetops import isin

from numpy.typing import ArrayLike, DTypeLike

class Generator:

    def __init__(self, chromossome_sizes:ArrayLike, chromossome_ranges:Union[None, List[Union[None, Tuple]], Tuple]=None, types:Union[list, DTypeLike]=[np.float32], names:Union[list, str, None]=None):    
        chromossome_sizes = np.asarray(chromossome_sizes)

        if chromossome_sizes.shape == ():
            chromossome_sizes = np.array([chromossome_sizes])

        n_chromossome = len(chromossome_sizes)

        if names is None:
            names = []
        elif isinstance(names, str):
            names = [names]
        

        if chromossome_ranges is None:
            chromossome_ranges = [None] * n_chromossome
        elif isinstance(chromossome_ranges, tuple):
            chromossome_ranges = [chromossome_ranges]

        if not isinstance(types, list):
            types = [types]

        self._names = []
        dtype = []
        for i in range(n_chromossome):
            name = "chr"+str(i)
            if len(names)-1 >= i:
                name = names[i]
            self._names.append(name)

            size = np.atleast_1d(chromossome_sizes[i])
            size = tuple(size)

            dtype.append((name, types[i], size))

            if chromossome_ranges[i] is None:
                if np.dtype(types[i]).char in np.typecodes["AllFloat"]:
                    chromossome_ranges[i] = (0.0, 1.0)
                elif np.dtype(types[i]).char in np.typecodes["AllInteger"]:
                    chromossome_ranges[i] = (0, 10)
        
        self._dtype = np.dtype(dtype)
        self._population = None
        self._chromossome_sizes = chromossome_sizes
        self._n_chromossome = n_chromossome
        self._chromossome_ranges = chromossome_ranges

    def generate_first(self, n_individual):

        population = np.empty(n_individual, self._dtype)

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


        