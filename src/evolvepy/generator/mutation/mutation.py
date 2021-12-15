from typing import Tuple, Union, List, Callable
from numba.misc.special import prange
from numba.np.ufunc import parallel
from numpy.typing import ArrayLike

import numpy as np
import numba


from .numeric_mutation import sum_mutation
from .binary_mutation import bit_mutation

from evolvepy.generator import Layer

def default_mutation(type):
    if (np.dtype(type).char in np.typecodes["AllFloat"] or 
        np.dtype(type).char in np.typecodes["AllInteger"]):
        return sum_mutation
    else:
        return bit_mutation

class NumericMutationLayer(Layer):
    def __init__(self, mutation_function:Callable, existence_rate:float, gene_rate:float, range:Tuple[float, float], name: str = None, chromossome_names: Union[str, List[str], None] = None):
        super().__init__(name=name, dynamic_parameters=True, chromossome_names=chromossome_names)
        self._mutation_function = mutation_function
        self._existence_rate = existence_rate
        self._gene_rate = gene_rate
        self._range = range

    def call(self, chromossomes: np.ndarray) -> np.ndarray:
        return NumericMutationLayer.mutate(chromossomes, self._mutation_function, self._existence_rate, self._gene_rate, self._range)

    @staticmethod
    @numba.njit(parallel=True)
    def mutate(chromossome:np.ndarray, mutation_function:Callable, existence_rate:float, gene_rate:float, range:Tuple[float, float]):
        result = np.empty_like(chromossome)
        
        for i in prange(chromossome.shape[0]):
            result[i] = mutation_function(chromossome[i], existence_rate, gene_rate, range)

        return result
