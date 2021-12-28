import numpy as np
from numpy.typing import ArrayLike
import numba

@numba.njit
def mean(chromosomes:ArrayLike) -> np.ndarray:
    '''Crossover computing the mean of chromosomes

        :param chromosomes: array of chromosomes
        :type chromosomes: np.typing.ArrayLike

        :return: new chromosome
        :rtype: np.ndarray
    '''

    chromosomes = np.asarray(chromosomes)

    return np.sum(chromosomes, axis=0)/chromosomes.shape[0]
    
@numba.njit
def one_point(chromosomes:ArrayLike) -> np.array:
    '''Crossover joining in one point

        :param chromosomes: array of chromosomes
        :type chromosomes: np.typing.ArrayLike

        :return: new chromosome
        :rtype: np.ndarray
    '''

    index = np.random.randint(chromosomes.shape[1])

    new_chromossome = np.empty_like(chromosomes[0])
    new_chromossome[:index] = chromosomes[0][:index]
    new_chromossome[index:] = chromosomes[1][index:]

    return new_chromossome

@numba.njit
def n_point(chromosomes:ArrayLike, n:int=1) -> np.array:
    '''Crossover joining in n points

        :param chromosomes: array of chromosomes
        :type chromosomes: np.typing.ArrayLike
        :param n: number of points to join
        :type n: int

        :return: new chromosome
        :rtype: np.ndarray
    '''
    indexs = np.random.randint(0, chromosomes.shape[1], size=n)

    indexs = np.sort(indexs)

    new_chromossome = np.empty_like(chromosomes[0])

    ant = 0

    ind_index = 0

    for i in range(n):
        new_chromossome[ant:indexs[i]] = chromosomes[ind_index][ant:indexs[i]]
        ant = indexs[i]

        if ind_index == 0:
            ind_index = 1
        else:
            ind_index = 0
    
    return new_chromossome

def default_crossover(type):
    return one_point

