from evolvepy.integrations.tf_keras import ProcessTFKerasFitnessFunction, EvolutionaryModel, ProcessTFKerasEvaluator
import evolvepy as ep
from tensorflow import keras
import numpy as np


class TestFunction(ProcessTFKerasFitnessFunction):

    def setup(self) -> None:
        return

    def evaluate(self, model: keras.Model) -> np.ndarray:
        x = np.zeros((1,1))
        y = model(x)[0][0].numpy()
        return y

if __name__ == "__main__":
    model = EvolutionaryModel([keras.layers.Dense(1, input_shape=(1,))])

    evaluator = ProcessTFKerasEvaluator(TestFunction, model, n_process=12)
    individuals = np.zeros(12, model.descriptor.dtype)

    del model
    keras.backend.clear_session()

    print(evaluator(individuals))
