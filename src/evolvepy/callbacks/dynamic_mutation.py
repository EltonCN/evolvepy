from typing import List
import numpy as np

from evolvepy.callbacks import Callback

class DynamicMutation(Callback):
    NORMAL = 0
    REFINEMENT = 1
    EXPLORATION = 2

    def __init__(self, layer_names:List[str], patience:int=10, refinement_patience:int=2, exploration_patience:int=2, refinement_steps:int=2, exploration_steps:int=5,  refinement_divider:int=2, exploration_multiplier:int=2):
        parameters = {}
        parameters["patience"] = patience
        parameters["refinement_patience"] = refinement_patience
        parameters["exploration_patience"] = exploration_patience
        parameters["refinement_steps"] = refinement_steps
        parameters["exploration_steps"] = exploration_steps
        parameters["refinement_divider"] = refinement_divider
        parameters["exploration_multiplier"] = exploration_multiplier
        parameters["wait"] = 0
        parameters["step_count"] = 0

        dynamic_parameters  = dict.fromkeys(list(parameters.keys()), True)

        dynamic_parameters["wait"] = False
        dynamic_parameters["step_count"] = False

        parameters["layer_names"] = layer_names
        self._layer_names = layer_names

        super().__init__(parameters=parameters, dynamic_parameters=dynamic_parameters)
        self._best_fitness = -np.Infinity

        self._stage = DynamicMutation.NORMAL

        self._original_parameters = {}

    def on_evaluator_end(self, fitness: np.ndarray) -> None:
        self._patience = self.parameters["patience"]
        self._exploration_patience = self.parameters["exploration_patience"]
        self._refinement_patience = self.parameters["refinement_patience"]
        self._exploration_multiplier = self.parameters["exploration_multiplier"]
        self._refinement_divider = self.parameters["refinement_divider"]
        self._refinement_steps = self.parameters["refinement_steps"]
        self._exploration_steps = self.parameters["exploration_steps"]
        self._wait = self.parameters["wait"]
        self._step_count = self.parameters["step_count"]

        
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

        self._parameters["wait"] = self._wait
        self._parameters["step_count"] = self._step_count

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