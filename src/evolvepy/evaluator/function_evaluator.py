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
    '''
    Evaluates individuals using a simple function.
    '''

    PYTHON = 0
    JIT = 1
    NJIT = 2
    JIT_PARALLEL = 3
    NJIT_PARALLEL = 4

    def __init__(self, function:Callable[[np.ndarray], ArrayLike], n_scores:int=1, mode:int=NJIT, individual_per_call:int = 1) -> None:
        '''
        FunctionEvaluator constructor.

        Args:
            function (Callable[[np.ndarray], ArrayLike]): Function that will be used to evaluate individuals.
            n_scores (int, optional): Number of scores generated for each individual. Defaults to 1.
            mode (int, optional): In which compilation mode to use the fitness function. Defaults to FunctionEvaluator.NJIT.
                                One of the class constants can be used:
                                PYTHON: No JIT compilation
                                JIT: With Numba JIT compilation
                                NJIT: With Numba No Python mode.
                                JIT_PARALLEL: With JIT and parallel assessments.
                                NJIT_PARALLEL: With NJIT and parallel assessments.
            individual_per_call (int, optional): Number of individuals that are evaluated at each function call. Defaults to 1.
        '''
        super().__init__(n_scores, individual_per_call, other_parameters={"evaluation_function_name":function.__name__})

        if mode == JIT:
            self._function = numba.jit()(function)
            self._static_call = numba.jit()(FunctionEvaluator.static_call)
        elif mode == NJIT:
            self._function = numba.njit()(function)
            self._static_call = numba.njit()(FunctionEvaluator.static_call)
        elif mode == JIT_PARALLEL:
            self._function = numba.jit(parallel=True)(function)
            self._static_call = numba.jit(parallel=True)(FunctionEvaluator.static_call)
        elif mode == NJIT_PARALLEL:
            self._function = numba.njit(parallel=True)(function)
            self._static_call = numba.njit(parallel=True)(FunctionEvaluator.static_call)
        else:
            self._function = function
            self._static_call = FunctionEvaluator.static_call

        self._mode = mode
        

    def call(self, population: np.ndarray) -> np.ndarray:
        '''
        Evaluates a population using the fitness function.

        Args:
            population (np.ndarray): Population to be evaluated.

        Returns:
            np.ndarray: Population fitness.
        '''
        self._scores  = self._static_call(self._function, self._individual_per_call, self._n_scores, population)
        return self._scores

    @staticmethod
    def static_call(function:Callable, individual_per_call:int, n_scores:int, population:np.ndarray) -> np.ndarray:
        '''
        Call the fitness function in the desired mode.

        Static method to enable just-in-time compilation using Numba.

        Args:
            function (Callable): Function that will be used to evaluate individuals.
            individual_per_call (int): Number of individuals that are evaluated at each function call.
            n_scores (int): Number of scores generated for each individual.
            population (np.ndarray): Population to be evaluated

        Returns:
            np.ndarray: Population fitness
        '''
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