from typing import Dict

import wandb

from evolvepy.callbacks.logger import Logger

class WandbLogger(Logger):

    def __init__(self, name:str=None, project:str=None, entity:str=None, group:str=None, log_fitness: bool = True, log_generator: bool = True, log_evaluator: bool = True, log_scores: bool = False):
        super().__init__(log_fitness=log_fitness, log_population=False, log_generator=log_generator, log_evaluator=log_evaluator, log_scores=log_scores)

        self._project = project
        self._entity = entity
        self._group = group
        self._name = name

        self._run = None

    def save_static_log(self, log: Dict[str, Dict]) -> None:
        self._run = wandb.init(project=self._project, entity=self._entity,
                    group=self._group, name=self._name, 
                    config=log, job_type="ea_optimization")
    
    def save_dynamic_log(self, log: Dict[str, Dict]) -> None:
        wandb.log(log)

    def __del__(self):
        if self._run is not None:
            wandb.finish()