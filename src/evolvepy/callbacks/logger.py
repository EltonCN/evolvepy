from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from evolvepy.callbacks.callback import Callback
from evolvepy.evaluator.evaluator import EvaluationStage, Evaluator


class Logger(Callback, ABC):
    def __init__(self, log_fitness:bool=True, log_population:bool=False, log_generator:bool=True, log_evaluator:bool=True, log_scores:bool=False):
        parameters = {"log_fitness":log_fitness, "log_population":log_population, 
                    "log_generator":log_generator, "log_evaluator":log_evaluator, "log_scores":log_scores}
        
        super().__init__(parameters=parameters)

        self._dynamic_log = {}

    def _get_evaluator_static_parameters(self, evaluator_log:Dict[str,object], evaluator:Evaluator) -> None:
        name = evaluator.name
        static_parameters = evaluator.static_parameters()

        for key in static_parameters:
            evaluator_log[name+"/"+key] = static_parameters[key]

    def _get_evaluator_dynamic_parameters(self, evaluator_log:Dict[str, object], evaluator:Evaluator) -> None:
        name = evaluator.name
        dynamic_parameters = evaluator.dynamic_parameters()

        for key in dynamic_parameters:
            evaluator_log[name+"/"+key] = dynamic_parameters[key]

    def on_start(self) -> None:
        generator_log = self.generator.get_all_static_parameters()
        evaluator_log = {}

        evaluator = self.evaluator
        while isinstance(evaluator, EvaluationStage):
            self._get_evaluator_static_parameters(evaluator_log, evaluator)
            evaluator = evaluator._evaluator
        self._get_evaluator_static_parameters(evaluator_log, evaluator)

        log= {"generator":generator_log, "evaluator":evaluator_log}

        self.save_static_log(log)

    @abstractmethod
    def save_static_log(self, log:Dict[str, Dict]) -> None:
        ...

    def on_generator_end(self, population: np.ndarray) -> None:
        self._dynamic_log = {}

        if self.parameters["log_population"]:
            self._dynamic_log["population"] = population

    def on_evaluator_end(self, fitness: np.ndarray) -> None:
        if self.parameters["log_fitness"]:
            self._dynamic_log["fitness"] = fitness

        if self.parameters["log_generator"]:
            self._dynamic_log["generator"] = self.generator.get_all_dynamic_parameters()

        if self.parameters["log_evaluator"]:
            evaluator_log = {}

            evaluator = self.evaluator
            while isinstance(evaluator, EvaluationStage):
                self._get_evaluator_dynamic_parameters(evaluator_log, evaluator)
                evaluator = evaluator._evaluator
            self._get_evaluator_dynamic_parameters(evaluator_log, evaluator)

            self._dynamic_log["evaluator"] = evaluator_log

        if self.parameters["log_scores"]:
            self._dynamic_log["scores"] = self.evaluator.scores

        self.save_dynamic_log(self._dynamic_log)
        

    @abstractmethod
    def save_dynamic_log(self, log:Dict[str,Dict]) -> None:
        ...

class MemoryStoreLogger(Logger):
    def __init__(self, log_fitness: bool = True, log_population: bool = False, log_generator: bool = True, log_evaluator: bool = True, log_scores: bool = False):
        super().__init__(log_fitness=log_fitness, log_population=log_population, log_generator=log_generator, log_evaluator=log_evaluator, log_scores=log_scores)

        self._log = []
        self._config_log = {}

    def save_dynamic_log(self, log: Dict[str, Dict]) -> None:
        self._log.append(log)

    def save_static_log(self, log: Dict[str, Dict]) -> None:
        self._config_log = log

    @property
    def log(self) -> List[Dict[str, Dict]]:
        return self._log

    @property
    def config_log(self) -> Dict[str, Dict]:
        return self._config_log