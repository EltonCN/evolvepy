from numba.core.utils import chain_exception
import numpy as np
from numpy.typing import ArrayLike
import numba
from typing import Tuple
from evolvepy.integrations import nvtx

@nvtx.annotate(domain="evolvepy", category="generator_operator")
@numba.njit
def sum_mutation(chromosome:ArrayLike, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float]):
    '''
    It takes a chromosome and add a random value between the mutation range to its gene, then repeats the process with
    the given probability.

    Args:
        chromosome (np.ArrayLike): array of chromosomes
        existence_rate (float): probability of first mutation
        gene_rate (float): probability of another gene mutation
        mutation_range (Tuple[float, float]):

    Returns:
        new_cromosome (np.ArrayLike): new mutated individual
    '''
    chromosome = np.asarray(chromosome)
    new_chromosome = chromosome.copy()
    
    first = True
    count = 0
    if np.random.rand() < existence_rate:
        while (first or np.random.rand() < gene_rate) and count < chromosome.shape[0]:
            first = False

            index = np.random.randint(0, chromosome.shape[0])
            new_chromosome[index] = chromosome[index] + np.random.uniform(mutation_range[0], mutation_range[1])
            count += 1

    return new_chromosome

@nvtx.annotate(domain="evolvepy", category="generator_operator")
def mul_mutation(chromosome:ArrayLike, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float]):
    '''
    It takes a chromosome and multiply a random value between the mutation range to its gene, then repeats the process with
    the given probability.

    Args:
        chromosome (np.ArrayLike): array of chromosomes
        existence_rate (float): probability of first mutation
        gene_rate (float): probability of another gene mutation
        mutation_range (Tuple[float, float]):

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
            new_chromosome[index] = new_chromosome[index] * np.random.uniform(mutation_range[0], mutation_range[1])

    return new_chromosome