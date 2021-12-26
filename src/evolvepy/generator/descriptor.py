from typing import Optional, Union, Tuple, List

import numpy as np
from numpy.typing import ArrayLike

class Descriptor:

    def __init__(self, chromossome_sizes:Optional[ArrayLike], chromossome_ranges:Union[None, List[Union[None, Tuple]], Tuple]=None, types:Union[list, DTypeLike]=[np.float32], names:Union[list, str, None]=None):
        chromossome_sizes = np.asarray(chromossome_sizes)

        if chromossome_sizes.shape == ():
            chromossome_sizes = np.array([chromossome_sizes])

        n_chromossome = len(chromossome_sizes)

        if names is None:
            names = []
        elif isinstance(names, str):
            names = [names]
        

        if chromossome_ranges is None:
            chromossome_ranges = [None] * n_chromossome
        elif isinstance(chromossome_ranges, tuple):
            chromossome_ranges = [chromossome_ranges]

        if not isinstance(types, list):
            types = [types]

        self._chromossome_sizes = chromossome_sizes
        self._n_chromossome = n_chromossome
        self._chromossome_ranges = chromossome_ranges
        
        self._create_dtype_names_ranges(names, types)
    
    def _create_dtype_names_ranges(self, names, types):
        self._names = []

        dtype = []
        for i in range(self._n_chromossome):
            name = "chr"+str(i)
            if len(names)-1 >= i:
                name = names[i]
            self._names.append(name)

            size = np.atleast_1d(self._chromossome_sizes[i])
            size = tuple(size)

            dtype.append((name, types[i], size))

            if self._chromossome_ranges[i] is None:
                if np.dtype(types[i]).char in np.typecodes["AllFloat"]:
                    self._chromossome_ranges[i] = (0.0, 1.0)
                elif np.dtype(types[i]).char in np.typecodes["AllInteger"]:
                    self._chromossome_ranges[i] = (0, 10)

        self._dtype = np.dtype(dtype)

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def chromossome_names(self):
        return self._chromossome_ranges

    @property
    def chromossome_ranges(self):
        return self._chromossome_ranges