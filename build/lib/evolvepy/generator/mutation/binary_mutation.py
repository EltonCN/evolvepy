import numpy as np
from numpy.typing import ArrayLike
import numba
from typing import Tuple

@numba.njit
def bit_mutation(chromossome:ArrayLike, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float]):
    ''' it takes a number n of genes and ramdomicaly change n gene bits in a chromossome.
		If gener_rate = 1 this mutaions behaves like a flipbit mutation, else it is a bitstring mutaion
	'''
    chromossome = np.asarray(chromossome)
    new_chromossome = chromossome.copy()

    first = True

    if np.random.rand() < existence_rate:
        while first or np.random.rand() < gene_rate:
            first = False

            index = np.random.randint(0, chromossome.shape[0])
            new_chromossome[index] = 1 if chromossome[index] == 0 else 0
