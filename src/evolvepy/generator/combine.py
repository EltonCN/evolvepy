from typing import Union, List, Callable
from numba.misc.special import prange
from numba.np.ufunc import parallel

import numpy as np
import numba

from evolvepy.generator import Layer

class CombineLayer(Layer):
    def __init__(self, selection_function:Callable, crossover_function:Callable, n_combine:int=2, name: str = None, chromossome_names: Union[str, List[str], None] = None):
        super().__init__(name=name, dynamic_parameters=False, chromossome_names=chromossome_names)

        self._selection_function = selection_function
        self._crossover_function = crossover_function
        self._n_combine = n_combine

    def call(self, chromossomes: np.ndarray, fitness:np.ndarray) -> np.ndarray:
        return CombineLayer.combine(chromossomes, fitness, self._selection_function, self._crossover_function, self._n_combine)

    @staticmethod
    @numba.njit(parallel=True)
    def combine(chromossomes:np.ndarray, fitness:np.ndarray, selection_function:Callable, crossover_function:Callable, n_combine:int):
        result = np.empty_like(chromossomes)

        n = fitness.shape[0]
        for i in prange(n):
            selected_indexes = selection_function(fitness, n_combine)
            selected = np.empty((n_combine, chromossomes.shape[1]), dtype=chromossomes.dtype)

            for j in range(n_combine):
                selected[j] = chromossomes[selected_indexes[j]]


            result[i] = crossover_function(selected)

        return result