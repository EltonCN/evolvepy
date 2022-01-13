from typing import Optional, List

from evolvepy.callbacks import Callback
from evolvepy.generator import Block, FirstGenLayer

class IncrementalEvolution(Callback):
    '''
    Callback that implements the behavior of a incremental evolution.

    Incremental evolution is the process of evolving some aspects of the solution sought at each moment.

    It works by preventing some pieces of the individuals from being altered, 
    allowing the adjustment of a part of them, usually that is more essential to the problem 
    (e.g. learning to process the readings of a robot's sensors before getting around)

    The piece of individuals that will be blocked must be on its own chromosome.

    This callback works in conjunction with two layers:
        - Block layer: prevents the chromosome to be changed. It must be in the generator before the layers that can alter the chromosome.
        - FirstGenLayer: generates the random distribution of the chromosome after its unlocking
    
    See the incremental evolution example to better understand how to use this callback: 
    https://github.com/EltonCN/evolvepy/blob/main/examples/Incremental%20Evolution.ipynb
    '''

    def __init__(self, generation_to_start:int, block_layer:Block, first_gen_layer:FirstGenLayer, callbacks:Optional[List[Callback]]=None):
        '''
        IncrementalEvolution constructor.

        Args:
            generation_to_start (int): In which generation to unlock the chromosome.
            block_layer (Block): Layer that will prevent the chromosome to be changed.
            first_gen_layer (FirstGenLayer): Layer that will generate the random distribution of the chromosome after its unlocking
            callbacks (List[Callback], optional): Callbacks that will be disabled along the chromosome. Defaults to None.
        '''
        parameters = {"generation_to_start":generation_to_start, "block_layer_name":block_layer.name, "first_gen_layer_name":first_gen_layer.name, "callbacks":[]}

        if callbacks is not None:
            for callback in callbacks:
                parameters["callbacks"].append(callback.name)
        else:
            callbacks = []

        super().__init__(parameters=parameters)

        self._generation = 0
        self._block_layer = block_layer
        self._first_gen_layer = first_gen_layer
        self._callbacks_to_stop = callbacks

    def on_generator_start(self) -> None:
        '''
        Called after generator run, performs the incremental evolution logic.

        See class documentation.
        '''
        if self._generation == self.parameters["generation_to_start"]:
            self._block_layer.parameters["run"] = False
            self._first_gen_layer.parameters["run"] = True

        elif self._generation > self.parameters["generation_to_start"]:
            self._block_layer.parameters["run"] = False
            self._first_gen_layer.parameters["run"] = False

            for callback in self._callbacks:
                callback.parameters["run"] = True 
        else:
            self._block_layer.parameters["run"] = True
            self._first_gen_layer.parameters["run"] = False

            for callback in self._callbacks_to_stop:
                callback.parameters["run"] = False 

        self._generation += 1