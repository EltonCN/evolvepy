import numpy as np

from evolvepy.evaluator.aggregator import FitnessAggregator
from evolvepy.evaluator.dispatcher import EvaluationDispatcher
from evolvepy.evaluator.evaluator import Evaluator

class EvaluationManager(Evaluator):
    def __init__(self, evaluator:Evaluator, dispatcher:EvaluationDispatcher=None, aggregator:FitnessAggregator=None):
        self._evaluator = evaluator
        self._dispatcher = dispatcher
        self._aggregator = aggregator

    def __call__(self, population: np.ndarray) -> np.ndarray:
        fitness = self._dispatcher(population, self._evaluator)
        fitness = self._aggregator(fitness)

        return fitness