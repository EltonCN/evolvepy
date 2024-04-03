from typing import List, Union

import numpy as np

from evolvepy.callbacks import Callback

class DynamicMutation(Callback):
    '''
    Callback that implements the behavior of a dynamic mutation.

    Dynamic mutation is the process of changing mutation ranges during the process of evolution, 
    to prevent the population from stagnating at a local maximum, 
    or never adjusting correctly to that maximum.

    It works in three stages, in order:

    Normal: Mutation occurs without changes.
    Refinement: Mutation decreases gradually, to correctly adjust to a maximum. It takes place in several steps.
    Exploration: Mutation gradually increases, to look for other maximums. It takes place in several steps.
                 It is stopped earlier if there is an improvement in fitness (it is understood that another local maximum has been found)

    A stage transition occurs when the best fitness does not change after a few generations.

    '''
    NORMAL = 0
    REFINEMENT = 1
    EXPLORATION = 2

    def __init__(self, layer_names:List[str], threshould:float=0, patience:int=10, refinement_patience:int=2, exploration_patience:int=2, refinement_steps:int=2, exploration_steps:int=5,  
                 refinement_divider:int=2, exploration_multiplier:int=2, stop_refinement:bool=False, watch_genes=False, run:bool=True):
        '''
        DynamicMutation constructor

        Args:
            layer_names (List[str]): Names of the NumericMutation layers that will be affected by this callback.
            threshould (float, optional): Minimum difference between generations to consider changing.
            patience (int, optional): How many generations in normal mode to wait before starting a transition (to refinement). Defaults to 10.
            refinement_patience (int, optional): How many generations in refinement mode to wait before starting a transition (to refinement or exploration). Defaults to 2.
            exploration_patience (int, optional): How many generations in exploration mode to wait before starting a transition (to exploration or normal). Defaults to 2.
            refinement_steps (int, optional): How many refinement steps will be performed. Defaults to 2.
            exploration_steps (int, optional): How many exploration steps will be performed. Defaults to 5.
            refinement_divider (int, optional): How much to divide the mutation rates at each refinement step. Defaults to 2.
            exploration_multiplier (int, optional): How much to multiply the mutation rates at each exploration step. Defaults to 2.
            stop_refinement (bool, optional): Whether to stop refining if you find an improvement in fitness. Defaults to False.
            watch_genes (bool, optional): If true, watchs for change in the best individual genes instead of the fitness. Defaults to False.
            run (bool, optional): Whether this callback should be executed. Defaults to True.

        Raises:
            ValueError: raised if layer_names is not a list.
        '''
        
        parameters = {}
        parameters["threshould"] = threshould
        parameters["patience"] = patience
        parameters["refinement_patience"] = refinement_patience
        parameters["exploration_patience"] = exploration_patience
        parameters["refinement_steps"] = refinement_steps
        parameters["exploration_steps"] = exploration_steps
        parameters["refinement_divider"] = refinement_divider
        parameters["exploration_multiplier"] = exploration_multiplier
        parameters["wait"] = 0
        parameters["step_count"] = 0
        parameters["watch_genes"] = watch_genes
        
        dynamic_parameters  = dict.fromkeys(list(parameters.keys()), True)

        dynamic_parameters["wait"] = False
        dynamic_parameters["step_count"] = False

        if not isinstance(layer_names, list):
            raise ValueError("layer_names must be a list of strings.")

        parameters["layer_names"] = layer_names
        parameters["stop_refinement"] = stop_refinement
        self._stop_refinement = stop_refinement
        self._layer_names = layer_names

        super().__init__(parameters=parameters, dynamic_parameters=dynamic_parameters, run=run)
        self._best_prev = -np.Infinity

        self._stage = DynamicMutation.NORMAL
        self._wait = 0

        self._original_parameters = {}


    def on_generator_end(self, population: np.ndarray) -> None:
        '''
        Saves population if is watching for gene variation.

        Args:
            population (np.ndarray): Population.
        '''

        if self.parameters["watch_genes"]:
            self._population = population

    def on_evaluator_end(self, fitness: np.ndarray) -> None:
        '''
        Called on evaluator end.

        Checks for maximum fitness (or best individual) and performs the dynamic mutation process as described in the class documentation

        Args:
            fitness (np.ndarray): Population fitness.
        '''

        self._patience = self.parameters["patience"]
        self._exploration_patience = self.parameters["exploration_patience"]
        self._refinement_patience = self.parameters["refinement_patience"]
        self._exploration_multiplier = self.parameters["exploration_multiplier"]
        self._refinement_divider = self.parameters["refinement_divider"]
        self._refinement_steps = self.parameters["refinement_steps"]
        self._exploration_steps = self.parameters["exploration_steps"]
        self._wait = self.parameters["wait"]
        self._step_count = self.parameters["step_count"]
        self._watch_genes = self.parameters["watch_genes"]
        self._threshould = self.parameters["threshould"]

        change, new_best = self._check_for_change(fitness)

        if change:
            self._best_prev = new_best
            self._wait = 0

            if self._stage == DynamicMutation.EXPLORATION or (self._stop_refinement and self._stage == DynamicMutation.REFINEMENT):
                #EXPLORATION -> NORMAL
                self._stage = DynamicMutation.NORMAL
                self._restore_parameters()
                self._step_count = 0
        else:
            self._wait += 1


        if self._stage == DynamicMutation.NORMAL and self._wait >= self._patience:
            self._wait = 0

            # NORMAL -> REFINEMENT
            self._stage = DynamicMutation.REFINEMENT
            self._step_count = 1
            self._save_parameters()
            self._refinement_step()
        elif self._stage == DynamicMutation.REFINEMENT and self._wait >= self._refinement_patience:
            self._wait = 0

            if self._step_count >= self._refinement_steps:
                # REFINEMENT -> EXPLORATION
                self._stage = DynamicMutation.EXPLORATION
                self._step_count = 1
                self._restore_parameters()
                self._exploration_step()
            else:
                # REFINEMENT
                self._refinement_step()
                self._step_count += 1
        elif self._stage == DynamicMutation.EXPLORATION and self._wait >= self._exploration_patience:
            self._wait = 0

            if self._step_count >= self._exploration_steps:
                # EXPLORATION -> NORMAL
                self._stage = DynamicMutation.NORMAL
                self._step_count = 0
                self._restore_parameters()
            else:
                # EXPLORATION
                self._exploration_step()
                self._step_count += 1

        self._parameters["wait"] = self._wait
        self._parameters["step_count"] = self._step_count

    def _check_for_change(self, fitness:np.ndarray) -> Union[float, np.ndarray]:
        '''
        Checks if the watched variable changed.

        Args:
            fitness (np.ndarray): current population fitness.

        Returns:
            Union[float, np.ndarray]: new best for the watched variable.
        '''

        change = False
        
        if self._watch_genes:
            bester_index = fitness.argmax()
            best_ind : np.ndarray = self._population[bester_index]
            new_best = best_ind

            if not isinstance(self._best_prev, float):
                chr_names = best_ind.dtype.names
                for chr_name in  chr_names:
                    if np.linalg.norm(self._best_prev[chr_name] - best_ind[chr_name]) > self._threshould:
                        change = True
            else:
                change = True
        else:
            new_best = fitness.max()
            change = np.abs(new_best-self._best_prev) > self._threshould

        return change, new_best


    def _save_parameters(self) -> None:
        '''
        Saves original parameters of mutation layers.
        '''
        for name in self._layer_names:
            self._original_parameters[name] = self.generator.get_parameters(name).copy()

    def _restore_parameters(self) -> None:
        '''
        Restores orginal parameters of mutation layers.
        '''
        for name in self._layer_names:
            self.generator.set_parameters(name, self._original_parameters[name])



    def _refinement_step(self):
        '''
        Performs a refinement step. Divides the mutation range.
        '''
        for name in self._layer_names:
            parameters = self.generator.get_parameters(name)

            if "mutation_range_min" in parameters:
                new_min = parameters["mutation_range_min"] / self._refinement_divider
                new_max = parameters["mutation_range_max"] / self._refinement_divider

                self.generator.set_parameter(name, "mutation_range_min", new_min)
                self.generator.set_parameter(name, "mutation_range_max", new_max)

    def _exploration_step(self):
        '''
        Performs a exploration step. Multiplys the mutation range.
        '''
        for name in self._layer_names:
            parameters = self.generator.get_parameters(name)

            if "mutation_range_min" in parameters:
                new_min = parameters["mutation_range_min"] * self._exploration_multiplier
                new_max = parameters["mutation_range_max"] * self._exploration_multiplier

                self.generator.set_parameter(name, "mutation_range_min", new_min)
                self.generator.set_parameter(name, "mutation_range_max", new_max)