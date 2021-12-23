from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np

from evolvepy.configurable import Configurable

class Evaluator(Configurable, ABC):
    
    def __init__(self, n_scores:int=1, individual_per_call:int = 1, other_parameters:Dict[str,object]=None, dynamic_parameters:Dict[str,bool]=None) -> None:
        if other_parameters is None:
            other_parameters={}
        
        other_parameters["n_scores"] = n_scores
        other_parameters["individual_per_call"] = individual_per_call

        super().__init__(other_parameters, dynamic_parameters)

        self._individual_per_call = individual_per_call
        self._n_scores = n_scores

    @abstractmethod
    def __call__(self, population:np.ndarray) -> np.ndarray:
        ...

class EvaluationStage(Evaluator):

    def __init__(self, evaluator:Evaluator, parameters:Dict[str, object]=None, dynamic_parameters:Dict[str, bool]=None) -> None:
        super().__init__(evaluator._n_scores, evaluator._individual_per_call, other_parameters=parameters, dynamic_parameters=dynamic_parameters)
        self._evaluator = evaluator
    

    def __call__(self, population:np.ndarray) -> np.ndarray:
        return self._evaluator(population)
