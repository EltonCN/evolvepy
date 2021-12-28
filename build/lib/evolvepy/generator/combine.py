from typing import Union, List, Callable

import numpy as np
import numba
from numba.misc.special import prange

from evolvepy.generator import ChromossomeOperator
from evolvepy.generator.context import Context

class CombineLayer(ChromossomeOperator):
    def __init__(self, selection_function:Callable, crossover_function:Callable, n_combine:int=2, name: str = None, chromossome_names: Union[str, List[str], None] = None):
        parameters = {"selection_function_name":selection_function.__name__, "crossover_function_name":crossover_function.__name__}
        super().__init__(name=name, chromossome_names=chromossome_names, parameters=parameters)

        self._selection_function = selection_function
        self._crossover_function = crossover_function
        self._n_combine = n_combine

    def call_chromossomes(self, chromossomes: np.ndarray, fitness:np.ndarray, context:Context) -> np.ndarray:
        return CombineLayer.combine(chromossomes, fitness, self._selection_function, self._crossover_function, self._n_combine)

    
    @staticmethod
    @numba.njit()#parallel=True)
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