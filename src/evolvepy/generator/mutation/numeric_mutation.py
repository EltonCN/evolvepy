import numpy as np
from numpy.typing import ArrayLike
import numba
from typing import Tuple


@numba.njit
def sum_mutation(chromossome:ArrayLike, existence_rate:float, gene_rate:float, range:Tuple[float, float]):
    new_chromossome = chromossome.copy()
    
    first = True

    if np.random.rand() < existence_rate:
        while first or np.random.rand() < gene_rate:
            first = False

            index = np.random.randint(0, chromossome.shape[0])
            new_chromossome[index] = chromossome[index] + np.random.uniform(range[0], range[1])

    return new_chromossome

def mul_mutation(chromossome:ArrayLike, existence_rate:float, gene_rate:float, range:Tuple[float, float]):
    new_chromossome = chromossome.copy()
    
    first = True

    if np.random.rand() < existence_rate:
        while first or np.random.rand() < gene_rate:
            first = False

            index = np.random.randint(0, chromossome.shape[0])
            new_chromossome[index] = new_chromossome[index] * np.random.uniform(range[0], range[1])

    return new_chromossome