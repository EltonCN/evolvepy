import numpy as np
import numba

from numpy.typing import ArrayLike

from .numeric_mutation import sum_mutation
from .binary_mutation import bit_mutation

def default_mutation(type):
    if (np.dtype(type).char in np.typecodes["AllFloat"] or 
        np.dtype(type).char in np.typecodes["AllInteger"]):
        return sum_mutation
    else:
        return bit_mutation