from __future__ import annotations

from typing import Dict, List
import numpy as np
from evolvepy.configurable import Configurable

from evolvepy.generator import Generator
from evolvepy.evaluator import Evaluator
from evolvepy.integrations import nvtx

class Callback(Configurable):
    '''
    Base Callback class.

    Callbacks are objects that can be called upon during evolution to change its behavior.
    '''

    def __init__(self, run:bool=True, parameters:Dict[str, object]=None, dynamic_parameters:Dict[str,bool]= None):
        '''
        Callback constructor.

        Args:
            run (bool, optional): Whether the object should run. Defaults to True.
            parameters (Dict[str, object], optional): Other callback parameters. Defaults to None.
            dynamic_parameters (Dict[str,bool], optional): Other callback dynamic parameters description. Defaults to None.
        '''
        
        if parameters is None:
            parameters = {}
        if dynamic_parameters is None:
            dynamic_parameters = {}
        parameters["run"] = run
        dynamic_parameters["run"] = run

        super().__init__(parameters, dynamic_parameters)
        
        self._generator : Generator = None
        self._evaluator : Evaluator = None
        self._callbacks : List[Callback] = []

    @property
    def generator(self) -> Generator:
        '''
        The Generator associated with evolution.

        Must be set correctly for Callback to work
        '''
        return self._generator
    
    @generator.setter
    def generator(self, value:Generator) -> None:
        if isinstance(value, Generator):
            self._generator = value
        else:
            raise ValueError("Generator must be a evolvepy Generator instance.")
    
    @property
    def evaluator(self) -> Evaluator:
        '''
        The Evaluator associated with evolution.

        Must be set correctly for Callback to work
        '''
        return self._evaluator
    
    @evaluator.setter
    def evaluator(self, value:Evaluator) -> None:
        if isinstance(value, Evaluator):
            self._evaluator = value
        else:
            raise ValueError("Evaluator must be a evolvepy Evaluator instance.")

    @property
    def callbacks(self) -> List[Callback]:
        '''
        Other callbacks associated with evolution

        Must be set correctly for Callback to work
        '''
        return self._callbacks
        
    @callbacks.setter
    def callbacks(self, value:List[Callback]) -> None:
        if not isinstance(value, list): 
            raise ValueError("callbacks must be a list")
        
        for callback in value:
            if not isinstance(callback, Callback):
                raise ValueError("All callbacks elements must be a evolvepy Callback instance.")
        
        self._callbacks = value

    #To call in Evolver

    def _on_start(self) -> None:
        '''
        Called when evolution start.
        '''
        range_name = "{0}_start".format(self.name)
        with nvtx.annotate_se(range_name, domain="evolvepy", category="callback", color=nvtx.callback_color):
            self.on_start()

    def _on_generator_start(self) -> None:
        '''
        Called before generator run.
        '''
        range_name = "{0}_generator_start".format(self.name)
        with nvtx.annotate_se(range_name, domain="evolvepy", category="callback", color=nvtx.callback_color):
            self.on_generator_start()
        

    def _on_generator_end(self, population:np.ndarray) -> None:
        '''
        Called after generator run, before evaluator.

        Args:
            population (np.ndarray): The generated population.
        '''
        range_name = "{0}_generator_end".format(self.name)
        with nvtx.annotate_se(range_name, domain="evolvepy", category="callback", color=nvtx.callback_color):
            self.on_generator_end(population)

    def _on_evaluator_end(self, fitness:np.ndarray) -> None:
        '''
        Called after evaluator run.

        Args:
            fitness (np.ndarray): The population fitness.
        '''
        range_name = "{0}_evaluator_end".format(self.name)
        with nvtx.annotate_se(range_name, domain="evolvepy", category="callback", color=nvtx.callback_color):
            self.on_evaluator_end(fitness)

    def _on_stop(self) -> None:
        '''
        Called on evolution end.
        '''
        range_name = "{0}_stop".format(self.name)
        with nvtx.annotate_se(range_name, domain="evolvepy", category="callback", color=nvtx.callback_color):
            self.on_stop()


    #To override
    def on_start(self) -> None:
        '''
        Called when evolution start.
        '''
        pass

    def on_generator_start(self) -> None:
        '''
        Called before generator run.
        '''
        pass

    def on_generator_end(self, population:np.ndarray) -> None:
        '''
        Called after generator run, before evaluator.

        Args:
            population (np.ndarray): The generated population.
        '''
        pass

    def on_evaluator_end(self, fitness:np.ndarray) -> None:
        '''
        Called after evaluator run.

        Args:
            fitness (np.ndarray): The population fitness.
        '''
        pass

    def on_stop(self) -> None:
        '''
        Called on evolution end.
        '''
        pass