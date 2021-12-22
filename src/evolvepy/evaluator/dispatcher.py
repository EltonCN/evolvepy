from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from evolvepy.evaluator.evaluator import Evaluator, EvaluationStage

        
class MultipleEvaluation(EvaluationStage):

    def __init__(self, evaluator:Evaluator, n_evaluation:int=1, agregator:Callable[[np.ndarray, int], np.ndarray]=np.mean) -> None:
        super().__init__(evaluator)
        self._n_evaluation = n_evaluation
        self._agregator = agregator
        self._n_scores = 1

    def __call__(self, population: np.ndarray) -> np.ndarray:
        fitness = np.empty((self._n_evaluation, len(population), self._evaluator._n_scores), dtype=np.float64)

        for i in range(self._n_evaluation):
            fitness[i] = self._evaluator(population)


        final_fitness = self._agregator(fitness, axis=0)

        return final_fitness