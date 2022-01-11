from abc import ABC
from typing import Dict, List, Union, Tuple

import numpy as np
import tensorflow.python.keras.layers


class Configurable(ABC):
    '''
        Base class for Configurable classes.
        
        Configurable elements classes have parameters that can be logged using EvolvePy's loggers.

        Attributes:
            __element_count (int): The number of created class instances. Is a class attribute.
    '''

    __element_count = 0

    @classmethod
    def reset_count(cls):
        '''
            Reset the class instances count.
        '''
        cls.__element_count = 0

    def __init__(self, parameters:Dict[str, object]=None, dynamic_parameters:Dict[str, bool]=None, name:str=None) -> None:
        '''
            Configurable constructor.

            Args:
                parameters (Dict[str, object]): Object logable parameters. 
                dynamic_parameters (Dict[str, bool]): Parameters that may change mid-run. 
                                                    The existence of a key indicates that it can change, 
                                                    its value indicates whether the change is locked (False) or not (True).

        '''
        
        if parameters is None:
            parameters = {}
        if dynamic_parameters is None:
            dynamic_parameters = {}
        
        if name is None:
            name = self.__class__.__name__
        self._name = name + str(Configurable.__element_count)
        Configurable.__element_count += 1
        
        self._parameters = parameters

        dynamic_parameter_names = list(dynamic_parameters.keys())
        parameters_names = list(parameters.keys())

        parameter_no_exist = np.in1d(dynamic_parameter_names, parameters_names, invert=True)
        if parameter_no_exist.sum() != 0:
            index = np.argwhere(parameter_no_exist)
            raise ValueError("Parameter "+dynamic_parameter_names[index]+" doesn't exist.")

        static_parameters = np.in1d(parameters_names, dynamic_parameter_names, invert=True)
        static_parameter_names = np.asarray(parameters_names)[static_parameters]
        static_parameter_names = list(static_parameter_names)

        self._static_parameter_names = static_parameter_names
        self._dynamic_parameter_names = dynamic_parameter_names

        self._dynamic_parameters = dynamic_parameters

    @property
    def parameters(self)-> Dict[str, object]:
        '''
            Object parameters.

            :setter:
            Args:
                value (Union[Dict[str, object], Tuple[str, object]]): If Dict[str, object], changes the value of all valid keys,
                                                                        if Tuple[str, object], changes the single key value, if valid.
        '''
        return self._parameters
    
    @parameters.setter
    def parameters(self, value:Union[Dict[str, object], Tuple[str, object]]) -> None:
        if isinstance(value, tuple):
            value = {value[0]: value[1]}
        
        keys = list(value.keys())
        for key in keys:
            if key not in self._dynamic_parameters or self._dynamic_parameters[key] == False:
                del value[key]

        self._parameters.update(value)

    @property
    def dynamic_parameters(self) -> Dict[str, object]:
        '''
            All dynamic parameters.
        '''
        return {key: self._parameters[key] for key in self._dynamic_parameter_names}

    @property
    def static_parameters(self) -> Dict[str, object]:
        '''
            All static parameters.
        '''
        return {key: self._parameters[key] for key in self._static_parameter_names}

    @property
    def name(self) -> str:
        '''
            Object name for logs.
        '''
        return self._name


    def lock_parameter(self, name:str) -> None:
        '''
            Lock some parameter, avoiding its change.

            Args:
                name (str): The parameter name.
        '''
        if name in self._dynamic_parameters:
            self._dynamic_parameters[name] = False
    
    def unlock_parameter(self, name:str) -> None:
        '''
            Unlock some parameter, allowing its change.

            Args:
                name (str): The parameter name.
        '''
        if name in self._dynamic_parameters:
            self._dynamic_parameters[name] = True