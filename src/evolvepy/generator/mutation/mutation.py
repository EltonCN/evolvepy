from typing import Tuple, Union, List, Callable, Optional
from numpy.typing import ArrayLike

import numpy as np
import numba

from evolvepy.generator.context import Context


from .numeric_mutation import sum_mutation
from .binary_mutation import bit_mutation

from evolvepy.generator import ChromosomeOperator

def default_mutation(type):
    if (np.dtype(type).char in np.typecodes["AllFloat"] or 
        np.dtype(type).char in np.typecodes["AllInteger"]):
        return sum_mutation
    else:
        return bit_mutation

class NumericMutationLayer(ChromosomeOperator):
    '''
    Layer destinated to apply the Numeric chromosome operations.
    '''

    def __init__(self, mutation_function:Callable, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float], name: str = None, chromosome_names: Union[str, List[str], None] = None):
        '''
        Generic caller to a mutation function passed as parameters.

        Args:
            mutation_function (class Callable): Define the function which will be used
            existence_rate (float): Probability of first mutation
            gene_rate (float): Probability of another gene mutation
            name (string): Name for the layer
            chromosome_names (Union[str, List[str], None]): Array of chromosomes names (optional)
        '''
        parameters = {"existence_rate":existence_rate, "gene_rate":gene_rate, "mutation_range_min":mutation_range[0], "mutation_range_max":mutation_range[1]}
        dynamic_parameters = dict.fromkeys(list(parameters.keys()), True)
        parameters["mutation_function_name"] = mutation_function.__name__

        super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters, chromosome_names=chromosome_names)
        self._mutation_function = mutation_function

    def call_chromosomes(self, chromosomes: np.ndarray, fitness:np.ndarray, context:Context, name:Optional[str]) -> np.ndarray:
        existence_rate = self.parameters["existence_rate"]
        gene_rate = self.parameters["gene_rate"]
        mutation_range = (self.parameters["mutation_range_min"], self.parameters["mutation_range_max"])

        return NumericMutationLayer.mutate(chromosomes, self._mutation_function, existence_rate, gene_rate, mutation_range)

    @staticmethod
    @numba.njit()#parallel=True)
    def mutate(chromosomes:np.ndarray, mutation_function:Callable, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float]):
        '''
        Generic caller to a mutation function passed as parameters.

        Args:
            chromosomes (np.ArrayLike): Array of chromosomes
            existence_rate (float): Probability of first mutation
            gene_rate (float): Probability of another gene mutation
            mutation_function (class Callable): Define the function which will be used

        Returns:
            result (np.ArrayLike): return a new mutated population
        '''
        result = np.empty_like(chromosomes)

        n = chromosomes.shape[0]
        for i in numba.prange(n):
            result[i] = mutation_function(chromosomes[i], existence_rate, gene_rate, mutation_range)

        return result


class BinaryMutationLayer(ChromosomeOperator):
    '''
    Layer destinated to apply the Binary chromosome operations.
    '''
    def __init__(self, mutation_function:Callable, existence_rate:float, gene_rate:float, name: str = None, chromosome_names: Union[str, List[str], None] = None):
        '''
        Generic caller to a mutation function passed as parameters.
            
        Args:
            mutation_function (class Callable): Define the function which will be used
            existence_rate (float): Probability of first mutation
            gene_rate (float): Probability of another gene mutation
            name (string): Name for the layer
            chromosome_names (Union[str, List[str], None]): Array of chromosomes names (optional)
        '''
        parameters = {"existence_rate":existence_rate, "gene_rate":gene_rate}
        dynamic_parameters = dict.fromkeys(list(parameters.keys()), True)
        parameters["mutation_function_name"] = mutation_function.__name__

        super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters, chromosome_names=chromosome_names)
        self._mutation_function = mutation_function

    def call_chromosomes(self, chromosomes: np.ndarray, fitness:np.ndarray, context:Context, name:Optional[str]) -> np.ndarray:
        '''
        Apply the mutation on the chromosomes

        Args:
            chromosomes (np.ArrayLike): Array of chromosomes
            fitness (np.array): Probability of first mutation
            context (class Context): Probability of another gene mutation
            name (string): Define the function which will be 

        Returns:
            BinaryMutationLayer.mutate: mutation function
        '''
        
        existence_rate = self.parameters["existence_rate"]
        gene_rate = self.parameters["gene_rate"]


        return BinaryMutationLayer.mutate(chromosomes, self._mutation_function, existence_rate, gene_rate)

    @staticmethod
    @numba.njit()
    def mutate(chromosomes:np.ndarray, mutation_function:Callable, existence_rate:float, gene_rate:float):
        '''
        Generic caller to a mutation function passed as parameters.

        Args:
            chromosomes (np.ArrayLike): array of chromosomes
            existence_rate (float): probability of first mutation
            gene_rate (float): probability of another gene mutation
            mutation_function (class Callable): Define the function which will be used

        Returns:
            result (np.ArrayLike): return a new mutated population
        '''
        result = np.empty_like(chromosomes)

        n = chromosomes.shape[0]
        for i in numba.prange(n):
            result[i] = mutation_function(chromosomes[i], existence_rate, gene_rate)

        return result
