import argparse
import time
from typing import Tuple

import numpy as np

from evolvepy.generator import Generator, Layer, Context, Concatenate, FilterFirsts, Descriptor
from evolvepy.integrations import nvtx

N_GENERATION = 10
POPULATION_SIZES = list(range(1, 21, 1))
DELAY_PER_INDIVIDUAL = 5E-2

class DelayLayer(Layer):
    def __init__(self, name: str = None, time_per_individual=1E-4):
        parameters = {"time_per_individual":time_per_individual}
        dynamic_parameters = {"time_per_individual":True}
        super().__init__(name, dynamic_parameters, parameters)

    def call(self, population:np.ndarray, fitness:np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
        delay = self.parameters["time_per_individual"]*population.shape[0]
        
        if delay > 0:
            time.sleep(delay)
        
        return population, fitness
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overhead", action="store_true", default=False)
    args = parser.parse_args()

    if args.overhead:
        DELAY_PER_INDIVIDUAL = 0

    population_size = 10
    
    li = Layer()

    l_flow1_1 = DelayLayer(time_per_individual=DELAY_PER_INDIVIDUAL)
    l_flow1_2 = FilterFirsts(np.ceil(population_size/2.0))

    l_flow2_1 = DelayLayer(time_per_individual=DELAY_PER_INDIVIDUAL)
    l_flow2_2 = FilterFirsts(np.floor(population_size/2.0))

    lf = Concatenate()

    li.next = l_flow1_1
    li.next = l_flow2_1

    l_flow1_1.next = l_flow1_2
    l_flow1_2.next = lf

    l_flow2_1.next = l_flow2_2
    l_flow2_2.next = lf

    descriptor = Descriptor()
    generator = Generator(first_layer=li, last_layer=lf, descriptor=descriptor)
    generator.generate(population_size)

    for population_size in POPULATION_SIZES:
        generator.set_parameter(l_flow1_2.name, "n_to_pass", int(np.ceil(population_size/2.0)))
        generator.set_parameter(l_flow2_2.name, "n_to_pass", int(np.floor(population_size/2.0)))

        for name in ["Serial", "Parallel"]:
            range_name = "{0}_{1}".format(name, population_size)
            with nvtx.annotate_se(range_name, category="benchmark", domain="evolvepy"):
                generator.generate(population_size)

