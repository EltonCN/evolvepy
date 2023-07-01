from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np

from evolvepy.configurable import Configurable
from evolvepy.integrations import nvtx

class Evaluator(Configurable, ABC):
    '''
    Base evaluator class.

    Must be inherited to be used.
    '''
    
    def __init__(self, n_scores:int=1, individual_per_call:int = 1, other_parameters:Dict[str,object]=None, dynamic_parameters:Dict[str,bool]=None) -> None:
        '''
        Evalutor constructor.

        Args:
            n_scores (int, optional): Number of scores generated when evaluating an individual. Defaults to 1.
            individual_per_call (int, optional): Number of individuals that are evaluated at each call. Defaults to 1.
            other_parameters (Dict[str,object], optional): Other parameters defined by the inheritor class. Defaults to None.
            dynamic_parameters (Dict[str,bool], optional): Other dynamic parameters defined by the inheritor class. Defaults to None.
        '''
        
        if other_parameters is None:
            other_parameters={}
        
        other_parameters["n_scores"] = n_scores
        other_parameters["individual_per_call"] = individual_per_call

        super().__init__(other_parameters, dynamic_parameters)

        self._individual_per_call = individual_per_call
        self._n_scores = n_scores
        self._scores : np.ndarray = None

    def __call__(self, population:np.ndarray) -> np.ndarray:
        '''
        Evaluates the population.

        Args:
            population (np.ndarray): Population to be evaluated.

        Returns:
            np.ndarray: Population fitness
        '''
        with nvtx.annotate_se(self.name, domain="evolvepy", category="evaluator", color=nvtx.evaluator_color):
            fitness = self.call(population)

        return fitness

    @property
    def scores(self) -> np.ndarray:
        '''
        Scores of the last evaluated individuals.
        '''
        return self._scores
    
    @abstractmethod
    def call(self, population:np.ndarray) -> np.ndarray:
        ...

class EvaluationStage(Evaluator):
    '''
    An evaluation stage. Allows you to modify the behavior of an evaluator.

    Can be chained with other stages.
    '''

    def __init__(self, evaluator:Evaluator, parameters:Dict[str, object]=None, dynamic_parameters:Dict[str, bool]=None) -> None:
        '''
        EvaluationStage constructor.

        Args:
            evaluator (Evaluator): Evaluator that will be modified.
            parameters (Dict[str, object], optional): Evaluator parameters. Defaults to None.
            dynamic_parameters (Dict[str, bool], optional): Evaluator dynamic parameters description. Defaults to None.
        '''
        super().__init__(evaluator._n_scores, evaluator._individual_per_call, other_parameters=parameters, dynamic_parameters=dynamic_parameters)
        self._evaluator = evaluator

        
    
    def call(self, population:np.ndarray) -> np.ndarray:
        return self._evaluator(population)