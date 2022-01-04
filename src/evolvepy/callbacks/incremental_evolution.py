from typing import Optional, List

from evolvepy.callbacks import Callback
from evolvepy.generator import Block, FirstGenLayer

class IncrementalEvolution(Callback):
    def __init__(self, generation_to_start:int, block_layer:Block, first_gen_layer:FirstGenLayer, callbacks:Optional[List[Callback]]=None):
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