from __future__ import annotations
from typing import Any, Dict, List, Union

class Context:

    default_values = ["sorted", "_sorted", "blocked", "_chromossome_names", "chromossome_names", "_values", "have_value", "copy", "_block_all", "block_all"]

    def __init__(self, chromossome_names:Union[List[str], None]=None, sorted=False):
        self._sorted = sorted

        if chromossome_names is None:
            self.blocked : bool = False
        else:
            self.blocked : Dict[str, bool] = dict.fromkeys(chromossome_names, False)

        self._chromossome_names = chromossome_names
        self._values : Dict[str, object] = {}
        self._block_all = False

    @property
    def chromossome_names(self) -> List[str]:
        return self._chromossome_names
    
    @property
    def sorted(self) -> bool:
        return self._sorted
    
    @sorted.setter
    def sorted(self, value:bool) -> None:
        if isinstance(value, bool):
            self._sorted = value
        else:
            raise ValueError("sorted must be a boolean")

    @property
    def block_all(self) -> bool:
        return self._block_all
    
    @block_all.setter
    def block_all(self, value:bool) -> None:
        if isinstance(value, bool):
            self._block_all = value
        else:
            raise ValueError("block_all must be a boolean")

    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in Context.default_values:
            super().__setattr__(__name, __value)
        else:
            self._values[__name] = __value

    def __getattribute__(self, __name: str) -> Any:
        if __name in ['__getstate__', '__setstate__'] + Context.default_values:
            return object.__getattribute__(self, __name)
        elif __name in self._values:
            return self._values[__name]
        else:
            raise AttributeError("Context doesn't have "+__name+" value")
    
    def have_value(self, name:str) -> bool:
        if name in self._values or name in Context.default_values:
            return True
        else:
            return False

    def copy(self) -> Context:
        context = Context(self.chromossome_names, self.sorted)
        
        if isinstance(self.blocked, bool):
            context.blocked = self.blocked
        else:
            context.blocked = dict(zip(self.blocked.keys(), self.blocked.values()))
        
        context._values = dict(zip(self._values.keys(), self._values.values()))

        return context
        
