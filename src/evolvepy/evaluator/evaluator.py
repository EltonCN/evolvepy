from abc import ABC, abstractmethod
from typing import Union

import numpy as np

class Evaluator(ABC):
    
    def __init__(self, n_scores:int=1, individual_per_call:int = 1) -> None:
        self._individual_per_call = individual_per_call
        self._n_scores = n_scores

    @abstractmethod
    def __call__(self, population:np.ndarray) -> np.ndarray:
        ...

class EvaluationStage(Evaluator):

    def __init__(self, evaluator:Evaluator) -> None:
        super().__init__(evaluator._n_scores)
        self._evaluator = evaluator
    

    def __call__(self, population:np.ndarray) -> np.ndarray:
        return self._evaluator(population)
