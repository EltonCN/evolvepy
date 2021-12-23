from typing import List
import numpy as np

from evolvepy.callbacks import Callback

class DynamicMutation(Callback):
    NORMAL = 0
    REFINEMENT = 1
    EXPLORATION = 2

    def __init__(self, layer_names:List[str], patience:int=10, refinement_patience:int=2, exploration_patience:int=2, refinement_steps:int=2, exploration_steps:int=5,  refinement_divider:int=2, exploration_multiplier:int=2):
        super().__init__()
        self._layer_names = layer_names
        self._patience = patience
        self._exploration_patience = exploration_patience
        self._refinement_patience = refinement_patience
        self._exploration_multiplier = exploration_multiplier
        self._refinement_divider = refinement_divider
        self._refinement_steps = refinement_steps
        self._exploration_steps = exploration_steps

        self._wait = 0
        self._step_count = 0
        self._best_fitness = -np.Infinity

        self._stage = DynamicMutation.NORMAL

        self._original_parameters = {}

    def on_evaluator_end(self, fitness: np.ndarray) -> None:
        
        max_fitness = fitness.max()
        if max_fitness > self._best_fitness:
            self._best_fitness = max_fitness
            self._wait = 0

            if self._stage == DynamicMutation.EXPLORATION:
                #EXPLORATION -> NORMAL
                self._stage = DynamicMutation.NORMAL
                self.restore_parameters()
                self._step_count = 0
        else:
            self._wait += 1


        if self._stage == DynamicMutation.NORMAL and self._wait >= self._patience:
            self._wait = 0

            # NORMAL -> REFINEMENT
            self._stage = DynamicMutation.REFINEMENT
            self._step_count = 1
            self.save_parameters()
            self.refinement_step()
        elif self._stage == DynamicMutation.REFINEMENT and self._wait >= self._refinement_patience:
            self._wait = 0

            if self._step_count >= self._refinement_steps:
                # REFINEMENT -> EXPLORATION
                self._stage = DynamicMutation.EXPLORATION
                self._step_count = 1
                self.restore_parameters()
                self.exploration_step()
            else:
                # REFINEMENT
                self.refinement_step()
                self._step_count += 1
        elif self._stage == DynamicMutation.EXPLORATION and self._wait >= self._exploration_patience:
            self._wait = 0

            if self._step_count >= self._exploration_steps:
                # EXPLORATION -> NORMAL
                self._stage = DynamicMutation.NORMAL
                self._step_count = 0
                self.restore_parameters()
            else:
                # EXPLORATION
                self.exploration_step()
                self._step_count += 1

    def save_parameters(self) -> None:
        for name in self._layer_names:
            self._original_parameters[name] = self.generator.get_parameters(name).copy()

    def restore_parameters(self) -> None:
        for name in self._layer_names:
            self.generator.set_parameters(name, self._original_parameters[name])



    def refinement_step(self):
        for name in self._layer_names:
            parameters = self.generator.get_parameters(name)

            if "mutation_range_min" in parameters:
                new_min = parameters["mutation_range_min"] / self._refinement_divider
                new_max = parameters["mutation_range_max"] / self._refinement_divider

                self.generator.set_parameter(name, "mutation_range_min", new_min)
                self.generator.set_parameter(name, "mutation_range_max", new_max)

    def exploration_step(self):
        for name in self._layer_names:
            parameters = self.generator.get_parameters(name)

            if "mutation_range_min" in parameters:
                new_min = parameters["mutation_range_min"] * self._exploration_multiplier
                new_max = parameters["mutation_range_max"] * self._exploration_multiplier

                self.generator.set_parameter(name, "mutation_range_min", new_min)
                self.generator.set_parameter(name, "mutation_range_max", new_max)