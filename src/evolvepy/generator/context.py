from __future__ import annotations
from typing import Any, Dict, List, Union

from evolvepy.integrations import nvtx

class Context:
    '''
    Layer to pass the context and restrictions from the previous layers
    '''
    default_values = ["sorted", "_sorted", "blocked", "_chromosome_names", "chromosome_names", "_values", "have_value", "copy", "_block_all", "block_all", "population_size", "_population_size"]

    def __init__(self, population_size:int, chromosome_names:Union[List[str], None]=None, sorted=False):
        '''
        Initialization for the context layer

        Args:
            population_size (int): Number of individuals
            chromosome_names (List[string]): Names of chromosomes
            sorted (bool): Flag to indicate f the population is sorted or not.
        '''
        self._sorted = sorted
        self._population_size = population_size

        if chromosome_names is None:
            self.blocked : bool = False
        else:
            self.blocked : Dict[str, bool] = dict.fromkeys(chromosome_names, False)

        self._chromosome_names = chromosome_names
        self._values : Dict[str, object] = {}
        self._block_all = False

    @property
    def population_size(self) -> int:
        '''
        Return population size
        
        Returns:
            population_size (int): population
        '''
        return self._population_size

    @property
    def chromosome_names(self) -> List[str]:
        '''
        Return chromosome names
        
        Returns:
            chromosome_names (int): chromosome names
        '''
        return self._chromosome_names
    
    @property
    def sorted(self) -> bool:
        '''
        Return if the population and fitness are sorted or not

        Setter:
            Raises:
                ValueError: raised if value is not a boolean.
        '''
        return self._sorted
    
    @sorted.setter
    def sorted(self, value:bool) -> None:
        if isinstance(value, bool):
            self._sorted = value
        else:
            raise ValueError("sorted must be a boolean")

    @property
    def block_all(self) -> bool:
        '''
        Prevents any alteration on all chromosomes

        Setter:
            Raises:
                ValueError: raised if block_all is not a boolean.
        '''
        return self._block_all
    
    @block_all.setter
    def block_all(self, value:bool) -> None:
        if isinstance(value, bool):
            self._block_all = value
        else:
            raise ValueError("block_all must be a boolean")

    
    def __setattr__(self, __name: str, __value: Any) -> None:
        '''
        Set the atribute
        '''
        if __name in Context.default_values:
            super().__setattr__(__name, __value)
        else:
            self._values[__name] = __value

    def __getattribute__(self, __name: str) -> Any:
        '''
        Returns the instance's attribute
        '''
        if __name in ['__getstate__', '__setstate__'] + Context.default_values:
            return object.__getattribute__(self, __name)
        elif __name in self._values:
            return self._values[__name]
        else:
            raise AttributeError("Context doesn't have "+__name+" value")
    
    def have_value(self, name:str) -> bool:
        '''
        Returns if name if within the registered values
        '''
        if name in self._values or name in Context.default_values:
            return True
        else:
            return False
    
    @nvtx.annotate(domain="evolvepy", category="generator")
    def copy(self) -> Context:
        '''
        Create a copy of this instance
        '''
        context = Context(self.population_size, self.chromosome_names, self.sorted)
        
        if isinstance(self.blocked, bool):
            context.blocked = self.blocked
        else:
            context.blocked = dict(zip(self.blocked.keys(), self.blocked.values()))
        
        context._values = dict(zip(self._values.keys(), self._values.values()))

        return context
        
