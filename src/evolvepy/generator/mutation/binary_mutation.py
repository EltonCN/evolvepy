import numpy as np
from numpy.typing import ArrayLike
import numba
from typing import Tuple

@numba.njit
def bit_mutation(chromosome:ArrayLike, existence_rate:float, gene_rate:float):
    ''' 
    It takes a number n of genes and randomicaly change n gene bits in a chromosome.
	If gene_rate = 1 this mutaions behaves like a flipbit mutation, else it is a bitstring mutaion.
    
    Args:
        chromosome (np.ArrayLike): list of chromosomes of an individual
        existentce_rate (float): maximum number of genes modified
        gene_rate (float): probability of gene mutation
    
    Returns:
        new_cromosome (np.ArrayLike): new mutated individual
	'''
    chromosome = np.asarray(chromosome)
    new_chromosome = chromosome.copy()

    first = True

    if np.random.rand() < existence_rate:
        while first or np.random.rand() < gene_rate:
            first = False

            index = np.random.randint(0, chromosome.shape[0])
            new_chromosome[index] = True if chromosome[index] == False else False
    
    return new_chromosome
