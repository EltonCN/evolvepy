from typing import Dict

import numpy as np

from .evaluator import Evaluator, EvaluationStage

class ShufflePopulation(EvaluationStage):
    '''
    Shuffles the population before evaluation.
    '''

    def __init__(self, evaluator: Evaluator) -> None:
        '''
        ShufflePopulation constructor.

        Args:
            evaluator (Evaluator): Evaluator used to evaluate the individuals.
        '''
        super().__init__(evaluator)
    
    def call(self, population: np.ndarray) -> np.ndarray:
        '''
        Evaluates the population, shuffling it before evaluation.

        The returned fitness are in the same order as 
        the incoming population.

        Args:
            population (np.ndarray): Population to be evaluated.

        Returns:
            np.ndarray: Population fitness.
        '''
        pop_size = len(population)

        indexes = list(range(pop_size))
        np.random.shuffle(indexes)

        shuffled_population = population[indexes]
        
        shuffled_fitness = self._evaluator(shuffled_population)

        fitness = np.empty_like(shuffled_fitness)
        fitness[indexes] = shuffled_fitness

        return fitness
