from typing import Any, Dict, Type
import multiprocessing as mp
from abc import ABC, abstractmethod

import numpy as np

from .evaluator import Evaluator

class ProcessFitnessFunction(ABC):

    def __init__(self, reset:bool=False, args:Any=None) -> None:
        self._setted = False
        self._reset = reset

    def __call__(self, individuals:np.ndarray) -> np.ndarray:
        if not self._setted or self._reset:
            self.setup()
            self._setted = True
        
        return self.evaluate(individuals)


    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def evaluate(self, individuals:np.ndarray) -> np.ndarray:
        ...
    
def evaluate_forever(fitness_function:Type[ProcessFitnessFunction], individuals_queue:mp.Queue, scores_queue:mp.Queue, args:Any):
    evaluator = fitness_function(args=args)

    while True:
        individuals, first, last = individuals_queue.get(block=True)
        scores = evaluator(individuals)
        scores_queue.put((scores, first, last))



class ProcessEvaluator(Evaluator):
    def __init__(self, fitness_function:Type[ProcessFitnessFunction], n_process:int=None, timeout:int=None, n_scores: int = 1, individual_per_call: int = 1, args:Any=None) -> None:
        if n_process is None:
            n_process = mp.cpu_count()
        
        other_parameters={"evaluation_function_name":fitness_function.__name__, "n_process":n_process, "timeout":timeout}
        super().__init__(n_scores=n_scores, individual_per_call=individual_per_call, other_parameters=other_parameters)

        self._fitness_function = fitness_function
        self._n_process = n_process
        self._timeout = timeout

        self._process = []
        self._individuals_queue = mp.Queue()
        self._scores_queue = mp.Queue()

        self._setted = False
        self._args = args


    def _prepare_process(self):
        if self._setted:
            return

        for _ in range(self._n_process):
            p = mp.Process(target=evaluate_forever, 
                            args=(self._fitness_function, self._individuals_queue, self._scores_queue, self._args),
                            daemon=True)
            p.start()
            self._process.append(True)
        
        self._setted = True


    def __call__(self, population: np.ndarray) -> np.ndarray:

        n = population.shape[0]//self._individual_per_call

        if n%self._individual_per_call != 0:
            raise RuntimeError("Population size must be divible by individual_per_call")

        self._prepare_process()

        for i in range(n):
            index = i*self._individual_per_call
            first = index
            last = index+self._individual_per_call

            individuals = population[first:last]

            self._individuals_queue.put((individuals, first, last))

        received = 0
        fitness = np.empty((population.shape[0], self._n_scores), dtype=np.float64)

        while received < n:
            scores, first, last = self._scores_queue.get(block=True, timeout=self._timeout)

            fitness[first:last] = np.asarray(scores).reshape((self._individual_per_call, self._n_scores))

            received += 1

        return fitness