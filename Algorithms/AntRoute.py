import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AntRoute:
    def __init__(self, pop_size: int, iterations: int, alpha: float, beta: float, rho: float, Graph, **kwargs):
        self.pop_size = pop_size
        self.iterations = iterations

        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.parts = np.random.randint(0, Graph.shape[0], self.pop_size)

    def run(self, **kwargs):
        raise NotImplementedError("Method not implemented")

    def plot(self):
        raise NotImplementedError("Method not implemented")


if __name__ == "__main__":
    graph = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    ar = AntRoute(10, 10, .5, .5, .5, graph)
