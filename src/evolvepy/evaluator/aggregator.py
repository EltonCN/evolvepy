from typing import Callable, List, Union
import numpy as np
from numpy.typing import ArrayLike

from evolvepy.evaluator.evaluator import EvaluationStage, Evaluator

class FitnessAggregator(EvaluationStage):
    '''
    Aggregates multiple individual scores into a final fitness.

    This stage must always be present in the case of evaluations with multiple fitness.
    '''
    MAX = 0
    MIN = 1
    MEAN = 2
    MEDIAN = 3

    MODE_NAMES = ["MAX", "MIN", "MEAN", "MEDIAN"]

    func :List[Callable] = [np.max, np.min, np.mean, np.median]

    def __init__(self, evaluator:Evaluator, mode:int = MEAN, weights:Union[ArrayLike, None] = None):
        '''
        FitnessAggregator constructor.

        Args:
            evaluator (Evaluator): Evaluator used to evaluate the individuals.
            mode (int, optional): How scores will be aggregated. Defaults to FitnessAggregator.MEAN.
                                Available modes: MAX, MIN, MEAN, MEDIAN.
            weights (Union[ArrayLike, None], optional): Weight to be used for each score. In the case of None, 
                                consider each score with the same weight. Defaults to None.
        '''
        if weights is not None:
            weights = np.asarray(weights)
        
        parameters = {"aggregation_mode":FitnessAggregator.MODE_NAMES[mode], "weights":weights}
        dynamic_parameters = {"weights":True}

        super().__init__(evaluator, parameters=parameters, dynamic_parameters=dynamic_parameters)
        
        self._mode = mode
        self._n_scores = 1

    def __call__(self, population:np.ndarray) -> np.ndarray:
        '''
        Evaluates a population aggregating the scores.

        Args:
            population (np.ndarray): Population to be evaluated.

        Returns:
            np.ndarray: Population fitness.
        '''
        fitness = self._evaluator(population)

        self._scores = fitness

        weights = self.parameters["weights"]
        if weights is not None:
            fitness = fitness*weights
        
        return FitnessAggregator.func[self._mode](fitness, axis=1).reshape((len(fitness), 1))
        