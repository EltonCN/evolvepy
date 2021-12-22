from typing import Any, Dict, List

class Context:

    default_values = ["sorted", "_sorted", "blocked", "_chromossome_names", "chromossome_names", "_values", "have_value"]

    def __init__(self, chromossome_names:List[str], sorted=False):
        self._sorted = sorted
        self.blocked : Dict[str, bool] = dict.fromkeys(chromossome_names, False)
        self._chromossome_names = chromossome_names
        self._values : Dict[str, object] = {}

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
            raise ValueError("Sorted must be a boolean")

    
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