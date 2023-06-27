import numpy as np

from evolvepy.evaluator import FitnessCache, FunctionEvaluator, MultipleEvaluation, FitnessAggregator, ProcessFitnessFunction, ProcessEvaluator
import evolvepy as ep
from evolvepy.generator import Generator, CombineLayer
from evolvepy.generator.descriptor import Descriptor
from evolvepy.generator.mutation import NumericMutationLayer, sum_mutation
from evolvepy.generator.crossover import one_point
from evolvepy.generator.selection import tournament
from evolvepy.evaluator import FunctionEvaluator
from evolvepy import Evolver


class ProcessSum(ProcessFitnessFunction):
    def __init__(self, args=None) -> None:
        super().__init__(reset=False)
    
    def setup(self) -> None:
        pass

    def evaluate(self, individuals: np.ndarray) -> np.ndarray:
        return individuals[0]["chr0"].sum()


if __name__ == "__main__":
    # Defines the layers of the generator
    layers = [CombineLayer(tournament, one_point), NumericMutationLayer(sum_mutation, 1.0, 0.0, (-10.0, 10.0))]

    # Specify that an individual has only one chromosome, which can vary between -1000 and 4000 
    descriptor = Descriptor(1, (-1000.0, 4000.0), [np.float32])

    # Creates the generator
    generator = Generator(layers=layers, descriptor=descriptor)

    process_evaluator = ProcessEvaluator(ProcessSum)

    # Here we specify for Evolver to use the previously created generator and evaluator, in generations of 100 individuals.
    evolver = Evolver(generator, process_evaluator, 100)

    hist, last_population = evolver.evolve(1000, verbose=True) 