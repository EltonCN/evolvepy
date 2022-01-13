from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from evolvepy.evaluator.evaluator import Evaluator, EvaluationStage

        
class MultipleEvaluation(EvaluationStage):
    '''
    Evaluates the same individual several times to avoid noise.
    '''

    def __init__(self, evaluator:Evaluator, n_evaluation:int=1, agregator:Callable[[np.ndarray, int], np.ndarray]=np.mean, discard_min=False, discard_max=False) -> None:
        '''
        MultipleEvaluation constructor.

        Args:
            evaluator (Evaluator): Evaluator used to evaluate the individuals.
            n_evaluation (int, optional): Number of evaluations to be carried out. Defaults to 1.
            agregator (Callable[[np.ndarray, int], np.ndarray], optional): Function that will aggregate the fitness of the evaluations. Defaults to np.mean.
            discard_min (bool, optional): Whether it should discard the lower fitness assessment. Defaults to False.
            discard_max (bool, optional): Whether it should discard the higher fitness assessment. Defaults to False.
        '''
        
        parameters = {"n_evaluation": n_evaluation, "agregator_name":agregator.__name__, "discard_min":discard_min, "discard_max":discard_max}

        super().__init__(evaluator, parameters, dynamic_parameters={"n_evaluation":True})
        self._agregator = agregator
        self._n_scores = 1
        self._discard_min = discard_min
        self._discard_max = discard_max

    def __call__(self, population: np.ndarray) -> np.ndarray:
        '''
        Evaluates the population several times, aggregating the evaluations

        Args:
            population (np.ndarray): Population to be evaluated.

        Returns:
            np.ndarray: Population aggregated fitness.
        '''

        n_evaluation = self.parameters["n_evaluation"]

        fitness = np.empty((n_evaluation, len(population), self._evaluator._n_scores), dtype=np.float64)

        for i in range(n_evaluation):
            fitness[i]  = self._evaluator(population)


        if self._discard_max or self._discard_min:
            result_size = n_evaluation
            if self._discard_max:
                result_size -= 1
            if self._discard_min:
                result_size -= 1

            result = np.empty((result_size, len(population), self._evaluator._n_scores))

            for i in range(len(population)):
                individual_fitness = fitness[:, i]
                individual_fitness = np.delete(individual_fitness, np.argmax(individual_fitness, axis=0), axis=0)
                individual_fitness = np.delete(individual_fitness, np.argmin(individual_fitness, axis=0), axis=0)
                result[:,i] = individual_fitness
                
        else:
            result = fitness

        final_fitness = self._agregator(result, axis=0)

        self._scores = final_fitness

        return final_fitness