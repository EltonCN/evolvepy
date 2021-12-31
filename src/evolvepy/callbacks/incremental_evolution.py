from evolvepy.callbacks import Callback
from evolvepy.generator import Block, FirstGenLayer

class IncrementalEvolution(Callback):
    def __init__(self, generation_to_start:int, block_layer:Block, first_gen_layer:FirstGenLayer):
        parameters = {"generation_to_start":generation_to_start, "block_layer_name":block_layer.name, "first_gen_layer_name":first_gen_layer.name}
        super().__init__(parameters=parameters)

        self._generation = 0
        self._block_layer = block_layer
        self._first_gen_layer = first_gen_layer

    def on_generator_start(self) -> None:
        if self._generation == self.parameters["generation_to_start"]:
            self._block_layer.parameters["run"] = False
            self._first_gen_layer.parameters["run"] = True

        elif self._generation > self.parameters["generation_to_start"]:
            self._block_layer.parameters["run"] = False
            self._first_gen_layer.parameters["run"] = False
        else:
            self._block_layer.parameters["run"] = True
            self._first_gen_layer.parameters["run"] = False 

        self._generation += 1