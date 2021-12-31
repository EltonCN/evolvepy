from typing import Tuple, Union, List, Callable, Optional
from numpy.typing import ArrayLike

import numpy as np
import numba

from evolvepy.generator.context import Context


from .numeric_mutation import sum_mutation
from .binary_mutation import bit_mutation

from evolvepy.generator import ChromossomeOperator

def default_mutation(type):
    if (np.dtype(type).char in np.typecodes["AllFloat"] or 
        np.dtype(type).char in np.typecodes["AllInteger"]):
        return sum_mutation
    else:
        return bit_mutation

class NumericMutationLayer(ChromossomeOperator):
    def __init__(self, mutation_function:Callable, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float], name: str = None, chromossome_names: Union[str, List[str], None] = None):
        parameters = {"existence_rate":existence_rate, "gene_rate":gene_rate, "mutation_range_min":mutation_range[0], "mutation_range_max":mutation_range[1]}
        dynamic_parameters = dict.fromkeys(list(parameters.keys()), True)
        parameters["mutation_function_name"] = mutation_function.__name__

        super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters, chromossome_names=chromossome_names)
        self._mutation_function = mutation_function

    def call_chromossomes(self, chromossomes: np.ndarray, fitness:np.ndarray, context:Context, name:Optional[str]) -> np.ndarray:
        existence_rate = self.parameters["existence_rate"]
        gene_rate = self.parameters["gene_rate"]
        mutation_range = (self.parameters["mutation_range_min"], self.parameters["mutation_range_max"])

        return NumericMutationLayer.mutate(chromossomes, self._mutation_function, existence_rate, gene_rate, mutation_range)

    @staticmethod
    @numba.njit()#parallel=True)
    def mutate(chromossomes:np.ndarray, mutation_function:Callable, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float]):
        result = np.empty_like(chromossomes)

        n = chromossomes.shape[0]
        for i in numba.prange(n):
            result[i] = mutation_function(chromossomes[i], existence_rate, gene_rate, mutation_range)

        return result
