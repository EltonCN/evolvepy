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

        self._running = False

        self._wandb_id = None

    def _start_wandb(self, log=None):
        if self._running:
            return

        if self._wandb_id is None:
            self._run = wandb.init(project=self._project, entity=self._entity,
                        group=self._group, name=self._name, 
                        config=log, job_type="ea_optimization", resume="allow")
        else:
            self._run = wandb.init(project=self._project, entity=self._entity,
                        group=self._group, name=self._name, 
                        config=log, job_type="ea_optimization", resume="allow", id=self._wandb_id)

        self._running = True
        self._wandb_id = self._run.id

    def save_static_log(self, log: Dict[str, Dict]) -> None:
        self._start_wandb(log=log)

    def save_dynamic_log(self, log: Dict[str, Dict]) -> None:
        self._start_wandb()

        wandb.log(log)

    def on_stop(self) -> None:
        super().on_stop()

        if self._running:
            wandb.finish()
            self._running = False

    def __del__(self):
        if self._running:
            wandb.finish()
            self._running = False