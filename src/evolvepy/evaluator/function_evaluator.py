from typing import Callable
from numba.misc.special import prange

from numba.np.ufunc import parallel
from .evaluator import Evaluator

import numpy as np
from numpy.typing import ArrayLike
import numba

PYTHON = 0
JIT = 1
NJIT = 2
JIT_PARALLEL = 3
NJIT_PARALLEL = 3

class FunctionEvaluator(Evaluator):

    PYTHON = 0
    JIT = 1
    NJIT = 2
    JIT_PARALLEL = 3
    NJIT_PARALLEL = 3

    def __init__(self, function:Callable[[np.ndarray], ArrayLike], n_scores:int=1, mode:int=NJIT, individual_per_call:int = 1) -> None:
        super().__init__(n_scores, individual_per_call, other_parameters={"evaluation_function_name":function.__name__})

        if mode == JIT:
            self._function = numba.jit()(function)
            self._static_call = numba.jit()(FunctionEvaluator.call)
        elif mode == NJIT:
            self._function = numba.njit()(function)
            self._static_call = numba.njit()(FunctionEvaluator.call)
        elif mode == JIT_PARALLEL:
            self._function = numba.jit(parallel=True)(function)
            self._static_call = numba.jit(parallel=True)(FunctionEvaluator.call)
        elif mode == NJIT_PARALLEL:
            self._function = numba.njit(parallel=True)(function)
            self._static_call = numba.njit(parallel=True)(FunctionEvaluator.call)
        else:
            self._function = function
            self._static_call = FunctionEvaluator.call

        self._mode = mode

    def __call__(self, population: np.ndarray) -> np.ndarray:
        return self._static_call(self._function, self._individual_per_call, self._n_scores, population)

    @staticmethod
    def call(function:Callable, individual_per_call:int, n_scores:int, population:np.ndarray) -> np.ndarray:
        n = population.shape[0]//individual_per_call

        #Can't raise exception with Numba
        #if n%individual_per_call != 0:
        #    raise RuntimeError("Population size must be divible by individual_per_call")

        fitness = np.empty((population.shape[0], n_scores), dtype=np.float64)

        for i in prange(n):
            index = i*individual_per_call
            first = index
            last = index+individual_per_call
            fitness[first:last] = np.asarray(function(population[first:last])).reshape((individual_per_call, n_scores))

        return fitness