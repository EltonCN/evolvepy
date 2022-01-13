[![](https://img.shields.io/pypi/v/evolvepy?style=for-the-badge)](https://pypi.org/project/evolvepy) [![](https://img.shields.io/pypi/l/evolvepy?style=for-the-badge)](https://github.com/EltonCN/evolvepy/blob/main/LICENSE) [![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/EltonCN/evolvepy)

# EvolvePy

EvolvePy is a Python module created to allow the easy creation and execution of evolutionary algorithms.

**Documentation**: [EvolvePy's documentation](https://eltoncn.github.io/evolvepy/_build/html/index.html).

Presentation video (Portuguese, click on the image):
[![Presentation Video](https://raw.githubusercontent.com/EltonCN/evolvepy/main/Thumbnail.png)](https://www.youtube.com/watch?v=Hfo1XkzEcis&list=PLXcCIm-yDGGfBveHj_f_31GkythkJpeUl&index=1)

## Features

(Links to example using the feature)

- Allows to create complex individual generators using different strategies:
  - [Crossover](https://github.com/EltonCN/evolvepy/blob/main/examples/Simple%20EA.ipynb) (one-point, n-point, mean)
  - Mutation ([sum](https://github.com/EltonCN/evolvepy/blob/main/examples/Simple%20EA.ipynb), multiplication, [binary](https://github.com/EltonCN/evolvepy/blob/main/examples/3-CNF-SAT.ipynb))
  - [Dynamic mutation](https://github.com/EltonCN/evolvepy/blob/main/examples/Dynamic%20Mutation.ipynb)
  - [Elitism](https://github.com/EltonCN/evolvepy/blob/main/examples/Elitism.ipynb)
  - [Randomic predation](https://github.com/EltonCN/evolvepy/blob/main/examples/Random%20Predation.ipynb)
  - [Incremental evolution](https://github.com/EltonCN/evolvepy/blob/main/examples/Incremental%20Evolution.ipynb)
- Define individuals with different chromosomes, with different types, ranges, sizes and parameters in the generator.
- Evaluate individuals using [simple functions](https://github.com/EltonCN/evolvepy/blob/main/examples/Simple%20EA.ipynb)  or [multiple processes](https://github.com/EltonCN/evolvepy/blob/main/examples/Car%20PID%20Control.ipynb).
  - [Fitness cache](https://github.com/EltonCN/evolvepy/blob/main/examples/Car%20PID%20Control.ipynb) to avoid evaluate the same individual several times.
  - Fitness functions with different scores, which can be aggregated with different strategies.
  - [Evaluate the same individual several times to avoid noise](https://github.com/EltonCN/evolvepy/blob/main/examples/Car%20PID%20Control.ipynb).
- [Log the evolution data to analyze later](https://github.com/EltonCN/evolvepy/blob/main/examples/Logger.ipynb).
- Integrations with other modules:
  - [Wandb](https://github.com/EltonCN/evolvepy/blob/main/examples/Logger.ipynb)
  - [Tensorflow/Keras](https://github.com/EltonCN/evolvepy/blob/main/examples/TF-Keras%20Integration.ipynb)
  - [Gym](https://github.com/EltonCN/evolvepy/blob/main/examples/Reinforcement%20Learning.ipynb)
  - [Unity ML Agents](https://github.com/EltonCN/evolvepy/blob/main/examples/Unity%20ML%20Agents%20-%203DBall.ipynb) (using Gym)

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


- For installing from the repository:

    ```bash
    git clone https://github.com/EltonCN/evolvepy
    cd evolvepy
    pip install --upgrade pip
    pip install .
    ```

## Examples

The ["examples"](https://github.com/EltonCN/evolvepy/blob/main/examples) folder have a lot of examples of how to use EvolvePy.

## Authors

Created by students from Unicamp's Institute of Computing (IC-Unicamp) as a project for the [evolutionary systems subject](https://gitlab.com/simoesusp/disciplinas/tree/master/SSC0713-Sistemas-Evolutivos-Aplicados-a-Robotica) at ICMC-USP, taught by prof. Eduardo do Valle Simoes.

- [EltonCN](https://github.com/EltonCN)
- [João Bonucci](https://github.com/Joao-Pedro-MB)
- [Thiago Lacerda](https://github.com/ThiagoDSL)