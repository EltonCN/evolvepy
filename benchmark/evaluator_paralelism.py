import argparse
import time

import numpy as np
import numba

from evolvepy.generator import Descriptor
from evolvepy.evaluator import FunctionEvaluator, ProcessEvaluator, ProcessFitnessFunction
from evolvepy.integrations import nvtx

N_WORKER = 4 
N_GENERATION = 10
POPULATION_SIZES = [1] + list(range(2, 21, 2)) 

def dummy_func(individuals:np.ndarray) -> float:
    return 0

def fitness_func(individuals:np.ndarray) -> float:
    with numba.objmode:
        time.sleep(1E-4)
    return 0

class ProcessFunc(ProcessFitnessFunction):
    def __init__(self, func) -> None:
        super().__init__()

        self.func = numba.njit()(func)

    def setup(self) -> None:
        return super().setup()

    def evaluate(self, individuals: np.ndarray) -> np.ndarray:
        return fitness_func(individuals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overhead", action="store_true", default=False)
    args = parser.parse_args()

    func = fitness_func
    if args.overhead:
        func = dummy_func

    numba.set_num_threads(N_WORKER)

    descriptor = Descriptor()
    
    evaluator_func = FunctionEvaluator(func, mode = FunctionEvaluator.NJIT, name="Base")
    evaluator_func_parallel = FunctionEvaluator(func, mode=FunctionEvaluator.NJIT_PARALLEL, name="Threads")
    evaluator_proc = ProcessEvaluator(ProcessFunc, n_process=4, args={"func": func}, name="Processes")
    
    evaluators = [evaluator_func, evaluator_func_parallel, evaluator_proc]

    for evaluator in evaluators:
        evaluator(np.empty(N_WORKER, descriptor.dtype))
    
    for population_size in POPULATION_SIZES:
        individuals = np.empty(population_size, descriptor.dtype)

        for evaluator in evaluators:

            range_name = "{0}_{1}".format(evaluator.name, population_size)
            for i in range(N_GENERATION):
                with nvtx.annotate_se(range_name, category="benchmark", domain="evolvepy"):
                    evaluator(individuals)


