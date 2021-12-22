from typing import Callable, List, Union
import numpy as np
from numpy.typing import ArrayLike

from evolvepy.evaluator.evaluator import EvaluationStage, Evaluator

class FitnessAggregator(EvaluationStage):

    MAX = 0
    MIN = 1
    MEAN = 2
    MEDIAN = 3

    func :List[Callable] = [np.max, np.min, np.mean, np.median]

    def __init__(self, evaluator:Evaluator, mode:int = MEAN, weights:Union[ArrayLike, None] = None):
        super().__init__(evaluator)
        self._mode = mode

        self._weights = None
        if weights is not None:
            self._weights = np.asarray(weights)

        self._n_scores = 1

    def __call__(self, population:np.ndarray) -> np.ndarray:
        fitness = self._evaluator(population)

        if self._weights is not None:
            fitness = fitness*self._weights
        
        return FitnessAggregator.func[self._mode](fitness, axis=1).reshape((len(fitness), 1))
        