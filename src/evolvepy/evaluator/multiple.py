from typing import Callable

import numpy as np
import numba

from evolvepy.evaluator.evaluator import Evaluator, EvaluationStage
from evolvepy.integrations import nvtx

        
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

    def call(self, population: np.ndarray) -> np.ndarray:
        '''
        Evaluates the population several times, aggregating the evaluations

        Args:
            population (np.ndarray): Population to be evaluated.

        Returns:
            np.ndarray: Population aggregated fitness.
        '''

        n_evaluation = self.parameters["n_evaluation"]

        fitness = np.empty((n_evaluation, len(population), self._evaluator._n_scores), dtype=np.float64)

        range_name = "{0}_iteration".format(self.name)
        for i in range(n_evaluation):
            with nvtx.annotate_se(range_name, domain="evolvepy", category="evaluator", color=nvtx.evaluator_color):
                fitness[i]  = self._evaluator(population)


        if self._discard_max or self._discard_min:
            range_name = "{0}_discard_maxmix".format(self.name)
            with nvtx.annotate_se(range_name, domain="evolvepy", category="evaluator", color=nvtx.evaluator_color):
                

                result = MultipleEvaluation._discard(population, fitness, self._evaluator._n_scores, n_evaluation, self._discard_min, self._discard_max)
        else:
            result = fitness

        range_name = "{0}_agregator".format(self.name)
        with nvtx.annotate_se(range_name, domain="evolvepy", category="evaluator", color=nvtx.evaluator_color):
            final_fitness = self._agregator(result, axis=0)

        self._scores = final_fitness

        return final_fitness
    
    @staticmethod
    @numba.njit(cache=True)
    def _discard(population, fitness, n_scores, n_evaluation, discard_min, discard_max):
        result_size = n_evaluation
        if discard_max:
            result_size -= 1
        if discard_min:
            result_size -= 1

        result = np.empty((result_size, len(population), n_scores))

        mask = np.empty(n_evaluation, np.bool_)
        for i in range(len(population)):
            mask[:] = True
            individual_fitness = fitness[:, i]

            if discard_max and discard_min:
                argmax = np.argmax(individual_fitness, axis=0)
                mask[argmax] = False
                
                argmin = np.argmin(individual_fitness[mask], axis=0)
                if argmin >= argmax:
                    argmin += 1
                mask[argmin] = False

            elif discard_max:
                mask[np.argmax(individual_fitness, axis=0)] = False
            elif discard_min:
                mask[np.argmin(individual_fitness, axis=0)] = False

            result[:,i] = individual_fitness[mask]
        
        return result