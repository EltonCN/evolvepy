from collections import deque
from typing import Deque, Dict, List

import numpy as np

from evolvepy.evaluator.evaluator import EvaluationStage, Evaluator
from evolvepy.integrations import nvtx

class FitnessCache(EvaluationStage):
    '''
    Evaluations cache. 
    
    It allows recovering the previous evaluation of an already evaluated individual, increasing the performance.
    '''

    def __init__(self, evaluator:Evaluator, n_generation:int = None, max_decimals:int=None):
        '''
        FitnessCache constructor.

        Args:
            evaluator (Evaluator): Evaluator used to evaluate the individuals.
            n_generation (int, optional): Amounts of generations to wait before deleting an old rating. Defaults to None (never deletes).
            max_decimals (int, optional): Decimal places to consider when comparing two individuals. Defaults to None (all places).
        '''
        parameters = {"n_generation":n_generation, "max_decimals":max_decimals}
        super().__init__(evaluator, parameters=parameters)
        
        self._n_generation = n_generation
        self._max_decimals = max_decimals
        self._cache : Dict[bytes, float] = {}
        self._first_acess : Dict[bytes, int] = {}
        self._generation = 0
        
    def get_individual_representation(self, individual:np.ndarray) -> bytes:
        '''
        Generates the byte representation of an individual, considering the precision in decimal places.

        Args:
            individual (np.ndarray): Individual who will have representation generated.

        Returns:
            bytes: Generated representation.
        '''
        
        if self._max_decimals is not None:
            if individual.dtype.names is not None:
                for name in individual.dtype.names:
                    individual[name] = np.round(individual[name], self._max_decimals)
            else:
                individual = np.round(individual, self._max_decimals)
            
        return individual.data.tobytes()


    def call(self, population:np.ndarray) -> np.ndarray:
        '''
        Returns the stored fitness of individuals, or evaluates them if it doesn't have it.

        Args:
            population (np.ndarray): Population to be evaluated.

        Returns:
            np.ndarray: Population fitness.
        '''
        pop_size = len(population)
        
        fitness = np.empty((pop_size, self._evaluator._n_scores), np.float64)
        to_evaluate_indexs = []
        to_evaluate_repr = []

        # Check for cache hit
        range_name = "{0}_hit_check".format(self.name)
        with nvtx.annotate_se(range_name, domain="evolvepy", category="evaluator"):
            for i in range(pop_size):
                
                range_name = "{0}_get_representation".format(self.name)
                with nvtx.annotate_se(range_name, domain="evolvepy", category="evaluator"):
                    ind_repr = self.get_individual_representation(population[i])

                if ind_repr not in self._cache:
                    to_evaluate_indexs.append(i)
                    to_evaluate_repr.append(ind_repr)
                else:
                    fitness[i] = self._cache[ind_repr]

                if ind_repr not in self._first_acess:
                    self._first_acess[ind_repr] = self._generation

        # Evaluate misses
        range_name = "{0}_miss_evaluation".format(self.name)
        with nvtx.annotate_se(range_name, domain="evolvepy", category="evaluator"):
            if len(to_evaluate_indexs) != 0:
                to_evaluate = population[to_evaluate_indexs]

                evaluated_fitness = self._evaluator(to_evaluate)

                fitness[to_evaluate_indexs] = evaluated_fitness

                for i in range(len(to_evaluate_repr)):
                    self._cache[to_evaluate_repr[i]] = evaluated_fitness[i]

        self._scores = self._evaluator.scores

        range_name = "{0}_delete_old".format(self.name)
        with nvtx.annotate_se(range_name, domain="evolvepy", category="evaluator"):
            self._delete_old()
        self._generation += 1

        return fitness

    def _delete_old(self) -> None:
        '''
        Delete old fitness.
        '''
        if self._n_generation is None:
            return

        for key in list(self._cache.keys()):
            if self._generation - self._first_acess[key] >= self._n_generation:
                del self._first_acess[key]
                del self._cache[key]