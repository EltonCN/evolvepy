from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from evolvepy.evaluator.evaluator import Evaluator

class EvaluationDispatcher:

    def __call__(self, population:np.ndarray, evaluator:Evaluator) ->np.ndarray:
        return evaluator(population)

        
class MultipleEvaluation(EvaluationDispatcher):

    def __init__(self, n_evaluation:int=1, agregator:Callable[[np.ndarray, int], np.ndarray]=np.mean) -> None:
        self._n_evaluation = n_evaluation
        self._agregator = agregator

    def __call__(self, population: np.ndarray, evaluator: Evaluator) -> np.ndarray:
        fitness = np.empty((self._n_evaluation, len(population), evaluator._n_scores), dtype=np.float64)

        for i in range(self._n_evaluation):
            fitness[i] = evaluator(population)


        final_fitness = self._agregator(fitness, axis=0)

        return final_fitness