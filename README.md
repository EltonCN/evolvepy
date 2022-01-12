# EvolvePy

EvolvePy is a Python module created to allow the easy creation and execution of evolutionary algorithms.

**Documentation**: [EvolvePy's documentation](https://eltoncn.github.io/evolvepy/_build/html/index.html).

## Features

(Links to example using feature)

- Allows to create complex individual generators using different strategies:
  - [Crossover](/examples/1%20-%20Simple%20EA.ipynb) (one-point, n-point, mean)
  - Mutation ([sum](/examples/1%20-%20Simple%20EA.ipynb), multiplication, binary)
  - Dynamic mutation
  - [Elitism](/examples/2%20-%20Elitism.ipynb)
  - [Randomic predation](/examples/6%20-%20Random%20Predation.ipynb)
  - [Incremental evolution](/examples/5%20-%20Incremental%20Evolution.ipynb)
- Define individuals with different chromosomes, with different types, ranges, sizes and parameters in the generator.
- Evaluate individuals using [simple functions](/examples/1%20-%20Simple%20EA.ipynb)  or [multiple processes](/examples/Car%20PID%20Control.ipynb).
  - [Fitness cache](/examples/Car%20PID%20Control.ipynb) to avoid evaluate the same individual several times.
  - Fitness functions with different scores, which can be aggregated with different strategies.
  - [Evaluate the same individual several times to avoid noise](/examples/Car%20PID%20Control.ipynb).
- [Log the evolution data to analyze later](/examples/4%20-%20Logger.ipynb).
- Integrations with other modules:
  - [Wandb](/examples/4%20-%20Logger.ipynb)
  - [Tensorflow/Keras](/examples/TF-Keras%20Integration.ipynb)
  - [Gym](/examples/Reinforcement%20Learning.ipynb)
  - [Unity ML Agents](/examples/Unity%20ML%20Agents%20-%203DBall.ipynb) (using Gym)

## Installation

- EvolvePy can be installed using pip:

    ```bash
    pip install --upgrade pip
    pip install evolvepy
    ```

- For install with all integrations dependecies (gym, tensorflow, wandb, gym_unity):
    
    ```bash
    pip install --upgrade pip
    pip install evolvepy[all_integrations]
    ```


- For installing from this repository:

    ```bash
    pip install --upgrade pip
    pip install .
    ```

## Examples

The ["examples"](/examples) folder have a lot of examples of how to use EvolvePy.

## Authors

Created by students from Unicamp's Institute of Computing (IC-Unicamp) as a project for the [evolutionary systems subject](https://gitlab.com/simoesusp/disciplinas/tree/master/SSC0713-Sistemas-Evolutivos-Aplicados-a-Robotica) at ICMC-USP, taught by prof. Eduardo do Valle Simoes.

- [EltonCN](https://github.com/EltonCN)
- [João Bonucci](https://github.com/Joao-Pedro-MB)
- [Thiago Lacerda](https://github.com/ThiagoDSL)