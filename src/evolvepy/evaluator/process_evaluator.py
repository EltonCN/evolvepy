from typing import Any, Dict, Type
import multiprocessing as mp
from abc import ABC, abstractmethod

import numpy as np

from .evaluator import Evaluator

class ProcessFitnessFunction(ABC):
    '''
    Base class of fitness function to be used to evaluate individuals in parallel with multiple processes.

    It must be inherited in a class that will implement the evaluation.
    '''

    def __init__(self, reset:bool=False) -> None:
        '''
        ProcessFitnessFunction constructor.

        Args:
            reset (bool, optional): Whether to reset the evaluator status at each evaluation. Defaults to False.
        '''
        self._setted = False
        self._reset = reset

    def __call__(self, individuals:np.ndarray) -> np.ndarray:
        '''
        Evaluates the individuals.

        Args:
            individuals (np.ndarray): Individuals who will be evaluated

        Returns:
            np.ndarray: Individuals fitness.
        '''
        if not self._setted or self._reset:
            self.setup()
            self._setted = True
        
        return self.evaluate(individuals)


    @abstractmethod
    def setup(self) -> None:
        '''
        Method that prepares the evaluator to evaluate individuals.

        Called before the first evaluation, and before all evaluations if "reset" is true.

        Must be implemented by the inheriting class.
        '''
        ...

    @abstractmethod
    def evaluate(self, individuals:np.ndarray) -> np.ndarray:
        '''
        Implements the evaluation of individuals.

        Must be implemented by the inheriting class.

        Args:
            population (np.ndarray): Individuals to be evaluated.

        Returns:
            np.ndarray: Individuals fitness.
        '''
        ...
    
def evaluate_forever(fitness_function:Type[ProcessFitnessFunction], individuals_queue:mp.Queue, scores_queue:mp.Queue, args:Dict[str, object]):
    '''
    Prepared the cost function to evaluate the individuals, waits for the receipt of individuals and returns the scores.

    Args:
        fitness_function (Type[ProcessFitnessFunction]): Class to be used to evaluate individuals.
        individuals_queue (mp.Queue): Queue in which individuals who need to be evaluated will arrive.
        scores_queue (mp.Queue): Queue in which the generated scores will be placed.
        args (Dict[str, object]): Other evaluator class constructor arguments.
    '''
    evaluator = fitness_function(**args)

    while True:
        individuals, first, last = individuals_queue.get(block=True)
        scores = evaluator(individuals)
        scores_queue.put((scores, first, last))



class ProcessEvaluator(Evaluator):
    '''
    Evaluates individuals using multiple process.

    '''

    def __init__(self, fitness_function:Type[ProcessFitnessFunction], n_process:int=None, timeout:int=None, n_scores: int = 1, individual_per_call: int = 1, args:Dict[str, object]=None) -> None:
        '''
        ProcessEvaluator constructor.

        Args:
            fitness_function (Type[ProcessFitnessFunction]): Class that will be used to evaluate individuals.
            n_process (int, optional): Number of process to use. Defaults to None (same number as cpu_count).
            timeout (int, optional): Maximum time to wait for a new evaluation. Defaults to None (infinity).
            n_scores (int, optional): Number of scores generated by fitness function. Defaults to 1.
            individual_per_call (int, optional): Number of individuals that the fitness function receives. Defaults to 1.
            args (Dict[str, object], optional): Other arguments for the fitness_function constructor. Defaults to None.
        '''
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

        if args is None:
            args = {}
        self._args = args


    def _prepare_process(self):
        '''
        Prepares the processes for evaluating individuals.
        '''
        if self._setted:
            return

        for _ in range(self._n_process):
            p = mp.Process(target=evaluate_forever, 
                            args=(self._fitness_function, self._individuals_queue, self._scores_queue, self._args),
                            daemon=True)
            p.start()
            self._process.append(p)
        
        self._setted = True


    def __call__(self, population: np.ndarray) -> np.ndarray:
        '''
        Evaluates the population

        Args:
            population (np.ndarray): Population to be evaluated.

        Raises:
            RuntimeError: Raised if population size is not divisible by individual_per_call.

        Returns:
            np.ndarray: Population fitness.
        '''

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