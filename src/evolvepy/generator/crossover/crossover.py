import numpy as np
from numpy.typing import ArrayLike
import numba

from evolvepy.integrations import nvtx

@numba.njit(cache=True)
def mean(chromosomes:ArrayLike) -> np.ndarray:
    '''
    Crossover computing the mean of chromosomes

    Args:
        chromosomes (np.typing.ArrayLike): array of chromosomes

    Returns:
		np.ndarray: new chromosome.
    '''

    chromosomes = np.asarray(chromosomes)

    return np.sum(chromosomes, axis=0)/chromosomes.shape[0]

@numba.njit(cache=True)
def one_point(chromosomes:ArrayLike) -> np.array:
    '''
    Crossover joining in one point
    Args:
        chromosomes (np.typing.ArrayLike): array of chromosomes

    Returns:
		np.ndarray: new chromosome.
    '''

    index = np.random.randint(chromosomes.shape[1])

    new_chromosome = np.empty_like(chromosomes[0])
    new_chromosome[:index] = chromosomes[0][:index]
    new_chromosome[index:] = chromosomes[1][index:]

    return new_chromosome

@numba.njit(cache=True)
def n_point(chromosomes:ArrayLike, n:int=1) -> np.array:
    '''
    Crossover joining in n points

    Args:
        chromosomes (np.typing.ArrayLike): array of chromosomes
        n (int): number of points to join

    Returns:
		np.ndarray: new chromosome.
    '''
    indexs = np.random.randint(0, chromosomes.shape[1], size=n)

    indexs = np.sort(indexs)

    new_chromosome = np.empty_like(chromosomes[0])

    ant = 0

    ind_index = 0

    for i in range(n):
        new_chromosome[ant:indexs[i]] = chromosomes[ind_index][ant:indexs[i]]
        ant = indexs[i]

        if ind_index == 0:
            ind_index = 1
        else:
            ind_index = 0
    
    return new_chromosome