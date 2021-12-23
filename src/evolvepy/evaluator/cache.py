from collections import deque
from typing import Deque, Dict, List

import numpy as np

from evolvepy.evaluator.evaluator import EvaluationStage, Evaluator

class FitnessCache(EvaluationStage):

    def __init__(self, evaluator:Evaluator, n_generation:int = None, max_decimals:int=None):
        parameters = {"n_generation":n_generation, "max_decimals":max_decimals}
        super().__init__(evaluator, parameters=parameters)
        
        self._n_generation = n_generation
        self._max_decimals = max_decimals
        self._cache : Dict[bytes, float] = {}
        self._last_acess : Dict[bytes, int] = {}
        self._generation = 0
        
    def get_individual_representation(self, individual:np.ndarray) -> bytes:
        if self._max_decimals is not None:
            if individual.dtype.names is not None:
                for name in individual.dtype.names:
                    individual[name] = np.round(individual[name], self._max_decimals)
            else:
                individual = np.round(individual, self._max_decimals)
            
        return individual.data.tobytes()


    def __call__(self, population:np.ndarray) -> np.ndarray:
        pop_size = len(population)
        
        fitness = np.empty((pop_size, self._evaluator._n_scores), np.float64)
        to_evaluate_indexs = []
        to_evaluate_repr = []

        # Check for cache hit
        for i in range(pop_size):

            ind_repr = self.get_individual_representation(population[i])

            if ind_repr not in self._cache:
                to_evaluate_indexs.append(i)
                to_evaluate_repr.append(ind_repr)
            else:
                fitness[i] = self._cache[ind_repr]

            if ind_repr not in self._last_acess:
                self._last_acess[ind_repr] = self._generation

        # Evaluate misses
        if len(to_evaluate_indexs) != 0:
            to_evaluate = population[to_evaluate_indexs]

            evaluated_fitness = self._evaluator(to_evaluate)

            fitness[to_evaluate_indexs] = evaluated_fitness

            for i in range(len(to_evaluate_repr)):
                self._cache[to_evaluate_repr[i]] = evaluated_fitness[i]

        self._scores = self._evaluator.scores

        self._delete_old()
        self._generation += 1

        return fitness

    def _delete_old(self) -> None:
        if self._n_generation is None:
            return

        for key in list(self._cache.keys()):
            if self._generation - self._last_acess[key] >= self._n_generation:
                del self._last_acess[key]
                del self._cache[key]