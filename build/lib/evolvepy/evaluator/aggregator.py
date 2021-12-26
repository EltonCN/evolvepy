from typing import Callable, List, Union
import numpy as np
from numpy.typing import ArrayLike

class FitnessAggregator:

    MAX = 0
    MIN = 1
    MEAN = 2
    MEDIAN = 3

    func :List[Callable] = [np.max, np.min, np.mean, np.median]

    def __init__(self, mode:int = MEAN, weights:Union[ArrayLike, None] = None):
        self._mode = mode

        self._weights = None
        if weights is not None:
            self._weights = np.asarray(weights)

    def __call__(self, fitness:np.ndarray) -> np.ndarray:
        if self._weights is not None:
            fitness = fitness*self._weights
        
        return FitnessAggregator.func[self._mode](fitness, axis=1)
        