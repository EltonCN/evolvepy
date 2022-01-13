from abc import ABC, abstractmethod
from typing import Dict, List, Union
import warnings
import datetime

import numpy as np

from evolvepy.callbacks.callback import Callback
from evolvepy.evaluator.evaluator import EvaluationStage, Evaluator


class Logger(Callback, ABC):
	'''
	Basic Logger callback class. 

	Allows to log data from the evolutionary process
	
	It needs to be inherited in some concrete save implementation to be used, 
	like MemoryStoreLogger, FileStoreLogger or WandbLogger.
	
	'''

	def __init__(self, log_fitness:bool=True, log_population:bool=False, log_generator:bool=True, log_evaluator:bool=True, log_scores:bool=False, log_best_individual:bool=True):
		'''
		Logger constructor.

		Args:
			log_fitness (bool, optional): Whether it should log the fitness of all individuals of each generation. Defaults to True.
			log_population (bool, optional):  Whether it should log the populations. Defaults to False.
			log_generator (bool, optional): Whether it should log the generator dynamic parameters. Defaults to True.
			log_evaluator (bool, optional): Whether it should log the evaluator dynamic parameters. Defaults to True.
			log_scores (bool, optional): Whether it should log all the evaluator individual scores. Defaults to False.
			log_best_individual (bool, optional): Whether it should log the best individual of each generation. Defaults to True.
		'''
		
		parameters = {"log_fitness":log_fitness, "log_population":log_population, 
					"log_generator":log_generator, "log_evaluator":log_evaluator, "log_scores":log_scores, "log_best_individual":log_best_individual}
		
		super().__init__(parameters=parameters)

		self._dynamic_log = {}
		self._generation_count = 0
		self._population = None

	def _get_evaluator_static_parameters(self, evaluator_log:Dict[str,object], evaluator:Evaluator) -> None:
		'''
		Adds the evaluator's static parameters to the log.

		Args:
			evaluator_log (Dict[str,object]): Log where the parameters will be added.
			evaluator (Evaluator): Evaluator where the parameters will be taken from.
		'''
		name = evaluator.name
		static_parameters = evaluator.static_parameters

		for key in static_parameters:
			evaluator_log[name+"/"+key] = static_parameters[key]

	def _get_evaluator_dynamic_parameters(self, evaluator_log:Dict[str, object], evaluator:Evaluator) -> None:
		'''
		Adds the evaluator's dynamic parameters to the log.

		Args:
			evaluator_log (Dict[str,object]): Log where the parameters will be added.
			evaluator (Evaluator): Evaluator where the parameters will be taken from.
		'''
		name = evaluator.name
		dynamic_parameters = evaluator.dynamic_parameters

		for key in dynamic_parameters:
			evaluator_log[name+"/"+key] = dynamic_parameters[key]

	def on_start(self) -> None:
		'''
		Runs on evolution start.

		Logs the static log.
		'''
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
		'''
		Saves static log.

		Must be implemented by the inheritor class.

		Args:
			log (Dict[str, Dict]): log with static parameters.
		'''
		...

	def on_generator_end(self, population: np.ndarray) -> None:
		'''
		Called on generator end.

		Adds the generated population to the log (if configured to do so), and the generation counter.

		Args:
			population (np.ndarray): Generated population that may be logged.
		'''
		self._dynamic_log = {}

		if self.parameters["log_population"]:
			self._dynamic_log["population"] = population
		self._population = population

		self._dynamic_log["generation"] = self._generation_count

	def on_evaluator_end(self, fitness: np.ndarray) -> None:
		'''
		Called after population evaluation.

		Adds the fitness, scores, dynamic parameters of generator and evaluator 
		and best individual if configured to do so. Also adds the best fitness.

		Saves the dynamic log.

		Args:
			fitness (np.ndarray): Population fitness.
		'''

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
		'''
		Saves dynamic log.

		Must be implemented by the inheritor class.

		Args:
			log (Dict[str, Dict]): log with static parameters.
		'''
		...


class MemoryStoreLogger(Logger):
	'''
	Logger that keeps all logged data in memory.

	Simple to use but can cause high memory usage.
	'''

	def __init__(self, log_fitness: bool = True, log_population: bool = False, log_generator: bool = True, log_evaluator: bool = True, log_scores: bool = False, log_best_individual:bool=True):
		'''
		MemoryStoreLogger constructor.

		Args:
			log_fitness (bool, optional): Whether it should log the fitness of all individuals of each generation. Defaults to True.
			log_population (bool, optional):  Whether it should log the populations. Defaults to False.
			log_generator (bool, optional): Whether it should log the generator dynamic parameters. Defaults to True.
			log_evaluator (bool, optional): Whether it should log the evaluator dynamic parameters. Defaults to True.
			log_scores (bool, optional): Whether it should log all the evaluator individual scores. Defaults to False.
			log_best_individual (bool, optional): Whether it should log the best individual of each generation. Defaults to True.
		'''
		super().__init__(log_fitness=log_fitness, log_population=log_population, log_generator=log_generator, log_evaluator=log_evaluator, log_scores=log_scores, log_best_individual=log_best_individual)

		self._log = []
		self._config_log = {}

	def save_dynamic_log(self, log: Dict[str, Dict]) -> None:
		'''
		Saves the dynamic log.

		Args:
			log (Dict[str, Dict]): Generation dynamic log to save.
		'''
		self._log.append(log)

	def save_static_log(self, log: Dict[str, Dict]) -> None:
		'''
		Saves the static log.

		Args:
			log (Dict[str, Dict]): Static log to save.
		'''
		self._config_log = log

	@property
	def log(self) -> List[Dict[str, Dict]]:
		'''
		Allows access to dynamic log

		Returns:
			List[Dict[str, Dict]]: The dynamic log.
		'''
		return self._log

	@property
	def config_log(self) -> Dict[str, Dict]:
		'''
		Allows access to static log

		Returns:
			List[Dict[str, Dict]]: The static log.
		'''
		return self._config_log


class FileStoreLogger(Logger):
	'''
	Logger that saves all logs in a file.
	'''
	def __init__(self, log_fitness: bool = True, log_population: bool = False, log_generator: bool = True, log_evaluator: bool = True, log_scores: bool = False, log_best_individual:bool=True):
		'''
		FileStoreLogger constructor.

		Args:
			log_fitness (bool, optional): Whether it should log the fitness of all individuals of each generation. Defaults to True.
			log_population (bool, optional):  Whether it should log the populations. Defaults to False.
			log_generator (bool, optional): Whether it should log the generator dynamic parameters. Defaults to True.
			log_evaluator (bool, optional): Whether it should log the evaluator dynamic parameters. Defaults to True.
			log_scores (bool, optional): Whether it should log all the evaluator individual scores. Defaults to False.
			log_best_individual (bool, optional): Whether it should log the best individual of each generation. Defaults to True.
		'''
		super().__init__(log_fitness=log_fitness, log_population=log_population, log_generator=log_generator, log_evaluator=log_evaluator, log_scores=log_scores, log_best_individual=log_best_individual)
		self._log_name : Union[None, str] = None

	def on_start(self) -> None:
		'''
		Called on evolution start.

		Prepare the log file.
		'''
		#setup logging basic configuration for logging to a file
		self._log_name = "./evolution_logs/debug_log_"+ str(datetime.datetime.now()) +".log"
		with open(self._log_name,"a") as debug_log:
			debug_log.close()

		super().on_start()

	def save_dynamic_log(self, log: Dict[str, Dict]) -> None:
		'''
		Dumps the dynamic log on the file.

		Args:
			log (Dict[str, Dict]): Log that will be dumped.
		'''
		with open(self._log_name,"a") as debug_log:
			debug_log.write('\n\n\n')
			for key, value in log.items():
				result = key + ': ' + str(value) + '\n'
				debug_log.write(result)
			

	def save_static_log(self, log: Dict[str, Dict]) -> None:
		'''
		Dumps the static log on the file.

		Args:
			log (Dict[str, Dict]): Log that will be dumped.
		'''
		with open(self._log_name,"a") as debug_log:
			debug_log.write('\n\n\n')
			for key, value in log.items():
				result = key + ': ' + str(value) + '\n'
				debug_log.write(result)
			
	@property
	def log_name(self) -> str:
		'''
		Name of the file where the log is saved
		'''
		if self._log_name is None:
			warnings.warn("Name of log file being accessed before logging starts. Returning null.", RuntimeWarning)

		return self._log_name