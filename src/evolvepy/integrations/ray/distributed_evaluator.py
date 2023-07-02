from typing import Any, Dict, Type
from abc import ABC, abstractmethod
import ray
from ray.util.queue import Queue
import numpy as np

from evolvepy.evaluator import Evaluator, ProcessFitnessFunction


@ray.remote
def evaluate_forever(fitness_function: Type[ProcessFitnessFunction], individuals_queue: Queue, scores_queue: Queue, args: Dict[str, object]):
    '''
    Prepares the cost function to evaluate the individuals, waits for the receipt of individuals, and returns the scores.

    Args:
        fitness_function (Type[ProcessFitnessFunction]): Class to be used to evaluate individuals.
        individuals_queue (Queue): Queue in which individuals who need to be evaluated will arrive.
        scores_queue (Queue): Queue in which the generated scores will be placed.
        args (Dict[str, object]): Other evaluator class constructor arguments.
    '''
    evaluator = fitness_function(**args)
    while True:
        individuals, first, last = individuals_queue.get()
        scores = evaluator(individuals)
        scores_queue.put((scores, first, last))


class DistributedEvaluator(Evaluator):
    def __init__(self, fitness_function: Type[ProcessFitnessFunction], n_worker: int = None, n_scores: int = 1, individual_per_call: int = 1, args: Dict[str, object] = None, name: str = None):
        '''
        Initializes the DistributedEvaluator.

        Args:
            fitness_function (Type[ProcessFitnessFunction]): Class to be used to evaluate individuals.
            n_worker (int, optional): Number of worker actors to create. Defaults to None.
            n_scores (int, optional): Number of scores per individual. Defaults to 1.
            individual_per_call (int, optional): Number of individuals to evaluate per actor call. Defaults to 1.
            args (Dict[str, object], optional): Other evaluator class constructor arguments. Defaults to None.
            name (str, optional): Name of the evaluator. Defaults to None.
        '''
        if n_worker is None:
            n_worker = int(ray.available_resources()["CPU"])

        other_parameters = {
            "evaluation_function_name": fitness_function.__name__, "n_worker": n_worker}
        super().__init__(n_scores=n_scores, individual_per_call=individual_per_call,
                         name=name, other_parameters=other_parameters)

        self._n_worker = n_worker
        self._fitness_function = fitness_function
        self._individuals_queue = Queue()
        self._scores_queue = Queue()

        if args is None:
            args = {}
        self._args = args

        self._setted = False

    def _prepare_actors(self):
        '''
        Prepares the actors for evaluation.
        '''
        if self._setted:
            return

        self._actors = [evaluate_forever.remote(self._fitness_function, self._individuals_queue, self._scores_queue, self._args)
                        for _ in range(self._n_worker)]

        self._setted = True

    def call(self, individuals: np.ndarray) -> np.ndarray:
        '''
        Evaluates the individuals using distributed evaluation.

        Args:
            individuals (np.ndarray): Array of individuals to evaluate.

        Returns:
            np.ndarray: Array of scores corresponding to the evaluated individuals.
        '''
        self._prepare_actors()

        n = individuals.shape[0] // self._individual_per_call

        first = 0
        last = 0
        for i in range(n):
            last = first + self._individual_per_call
            self._individuals_queue.put((individuals[first:last], first, last))
            first = last

        scores = np.empty(
            (individuals.shape[0], self._n_scores), dtype=np.float64)
        received = 0

        while received < n:
            scores_chunk, first, last = self._scores_queue.get()
            scores[first:last] = scores_chunk
            received += last - first
        return scores
