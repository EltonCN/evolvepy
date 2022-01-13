from typing import Union, List, Callable, Optional

import numpy as np
import numba
from numba.misc.special import prange

from evolvepy.generator import ChromosomeOperator
from evolvepy.generator.context import Context

class CombineLayer(ChromosomeOperator):
    '''
    Layer to join different pipeline paths
    '''
    def __init__(self, selection_function:Callable, crossover_function:Callable, n_combine:int=2, name: str = None, chromosome_names: Union[str, List[str], None] = None):
        '''
        Initialization for combine layer setting selection, crossover and n_combine parameters

        Args:
            selection_function (Callable): Selection function used fot the layer
            crossover_function (Callable): Crossover function used for the layer
            n_combine (int): Number of layers to be combined
            name (string): Name of the layer
            chromosome_names (List[string]): list of names from the chromosomes
        '''
        parameters = {"selection_function_name":selection_function.__name__, "crossover_function_name":crossover_function.__name__}
        super().__init__(name=name, chromosome_names=chromosome_names, parameters=parameters)

        self._selection_function = selection_function
        self._crossover_function = crossover_function
        self._n_combine = n_combine

    def call_chromosomes(self, chromosomes: np.ndarray, fitness:np.ndarray, context:Context, name:Optional[str]) -> np.ndarray:
        '''
        Generic call for combine function
        '''
        return CombineLayer.combine(chromosomes, fitness, self._selection_function, self._crossover_function, self._n_combine)

    
    @staticmethod
    @numba.njit()#parallel=True)
    def combine(chromosomes:np.ndarray, fitness:np.ndarray, selection_function:Callable, crossover_function:Callable, n_combine:int):
        '''
        Combine two or more layers

        Args:
            chromossomes (np.ndarray): Array of chromosomes
            fitness (np.ndarray): Array of inidividuals fitness
            selection_function (Callable): Selection function for the layer
            crossover_function (Callable): Crossover function for the layer
            n_combine (int): Number of layers to combine

        Returns:
            result (np.ArrayLike): Array of new population

        '''
        result = np.empty_like(chromosomes)

        n = fitness.shape[0]
        for i in prange(n):
            selected_indexes = selection_function(fitness, n_combine)
            selected = np.empty((n_combine, chromosomes.shape[1]), dtype=chromosomes.dtype)

            for j in range(n_combine):
                selected[j] = chromosomes[selected_indexes[j]]


            result[i] = crossover_function(selected)

        return result