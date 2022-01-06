from abc import ABC, abstractmethod
from typing import Dict, List
import datetime, json

import numpy as np

from evolvepy.callbacks.callback import Callback
from evolvepy.evaluator.evaluator import EvaluationStage, Evaluator


class Logger(Callback, ABC):
	def __init__(self, log_fitness:bool=True, log_population:bool=False, log_generator:bool=True, log_evaluator:bool=True, log_scores:bool=False, log_best_individual:bool=True):
		parameters = {"log_fitness":log_fitness, "log_population":log_population, 
					"log_generator":log_generator, "log_evaluator":log_evaluator, "log_scores":log_scores, "log_best_individual":log_best_individual}
		
		super().__init__(parameters=parameters)

		self._dynamic_log = {}
		self._generation_count = 0
		self._population = None

	def _get_evaluator_static_parameters(self, evaluator_log:Dict[str,object], evaluator:Evaluator) -> None:
		name = evaluator.name
		static_parameters = evaluator.static_parameters

		for key in static_parameters:
			evaluator_log[name+"/"+key] = static_parameters[key]

	def _get_evaluator_dynamic_parameters(self, evaluator_log:Dict[str, object], evaluator:Evaluator) -> None:
		name = evaluator.name
		dynamic_parameters = evaluator.dynamic_parameters

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
		self._population = population

		self._dynamic_log["generation"] = self._generation_count

	def on_evaluator_end(self, fitness: np.ndarray) -> None:
		fitness = fitness.flatten()
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

		best_index = np.argmax(fitness)

		self._dynamic_log["best_fitness"] = fitness[best_index]

		if self.parameters["log_best_individual"]:

			if self._population[0].dtype is None:
				self._dynamic_log["best_individual"] = self._population[best_index]
			else:
				for name in self._population[0].dtype.names:
					for i in range(len(self._population[0][name])):
						self._dynamic_log["best_individual/"+name+"/"+str(i)] = self._population[best_index][name][i]


		self.save_dynamic_log(self._dynamic_log)
		self._generation_count += 1
		
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


class FileStoreLogger(Logger):
	def __init__(self, log_fitness: bool = True, log_population: bool = False, log_generator: bool = True, log_evaluator: bool = True, log_scores: bool = False):
		super().__init__(log_fitness=log_fitness, log_population=log_population, log_generator=log_generator, log_evaluator=log_evaluator, log_scores=log_scores)

		#setup logging basic configuration for logging to a file
		self._log_name = "./evolution_logs/debug_log_"+ str(datetime.datetime.now()) +".log"
		with open(self._log_name,"a") as debug_log:
			debug_log.close()

	def save_dynamic_log(self, log: Dict[str, Dict]) -> None:
		with open(self._log_name,"a") as debug_log:
			debug_log.write('\n\n\n')
			for key, value in log.items():
				result = key + ': ' + str(value) + '\n'
				debug_log.write(result)
			

	def save_static_log(self, log: Dict[str, Dict]) -> None:
		with open(self._log_name,"a") as debug_log:
			debug_log.write('\n\n\n')
			for key, value in log.items():
				result = key + ': ' + str(value) + '\n'
				debug_log.write(result)
			
	@property
	def log(self) -> str:
		return "O log pode ser encontrado no arquivo " + self._log_name