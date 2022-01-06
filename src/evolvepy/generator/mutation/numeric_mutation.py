from numba.core.utils import chain_exception
import numpy as np
from numpy.typing import ArrayLike
import numba
from typing import Tuple


@numba.njit
def sum_mutation(chromossome:ArrayLike, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float]):
    chromossome = np.asarray(chromossome)
    new_chromossome = chromossome.copy()
    
    first = True
    count = 0
    if np.random.rand() < existence_rate:
        while (first or np.random.rand() < gene_rate) and count < chromossome.shape[0]:
            first = False

            index = np.random.randint(0, chromossome.shape[0])
            new_chromossome[index] = chromossome[index] + np.random.uniform(mutation_range[0], mutation_range[1])
            count += 1

    return new_chromossome

def mul_mutation(chromossome:ArrayLike, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float]):
    chromossome = np.asarray(chromossome)
    new_chromossome = chromossome.copy()
    
    first = True

    if np.random.rand() < existence_rate:
        while first or np.random.rand() < gene_rate:
            first = False

            index = np.random.randint(0, chromossome.shape[0])
            new_chromossome[index] = new_chromossome[index] * np.random.uniform(mutation_range[0], mutation_range[1])

    return new_chromossome