from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from evolvepy.evaluator.evaluator import Evaluator, EvaluationStage

        
class MultipleEvaluation(EvaluationStage):

    def __init__(self, evaluator:Evaluator, n_evaluation:int=1, agregator:Callable[[np.ndarray, int], np.ndarray]=np.mean) -> None:
        parameters = {"n_evaluation": n_evaluation, "agregator_name":agregator.__name__}

        super().__init__(evaluator, parameters, dynamic_parameters={"n_evaluation":True})
        self._agregator = agregator
        self._n_scores = 1

    def __call__(self, population: np.ndarray) -> np.ndarray:
        n_evaluation = self.parameters["n_evaluation"]

        fitness = np.empty((n_evaluation, len(population), self._evaluator._n_scores), dtype=np.float64)

        for i in range(n_evaluation):
            fitness[i] = self._evaluator(population)

        final_fitness = self._agregator(fitness, axis=0)

        self._scores = final_fitness

        return final_fitness