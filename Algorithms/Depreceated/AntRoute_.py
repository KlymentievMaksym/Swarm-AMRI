import numpy as np
# import random as rng
from tqdm import tqdm

import matplotlib.pyplot as plt

if __name__ == "__main__" or __name__ == "__mp_main__":
    # from AlgoritmRoute import AlgoritmRoute
    from Circle import Circle
    from Algorithms.AntTSP_ import ant_colony_tsp
else:
    # from .AlgoritmRoute import AlgoritmRoute
    from Circle import Circle
    from .AntTSP import ant_colony_tsp


class AntRoute:
    def __init__(self, pop_size: int, iterations: int, alpha: float, beta: float, rho: float, Q: int, graph: np.ndarray | dict = None, threads_count: int = 4, **kwargs):
        # super().__init__(pop_size, iterations, graph, **kwargs)
        self.pop_size = pop_size
        self.iterations = iterations

        self.graph = graph
        self.dim = len(self.graph)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.Q = Q

        # self.kwargs = kwargs
        self.history_best_f = np.zeros(self.iterations)
        self.history_best_routes = np.zeros((self.iterations, self.dim), dtype=int)

    def run(self, **kwargs):
        # self.prerun(**kwargs)
        # for iteration in tqdm(
        #     range(self.iterations),
        #     desc=f"Processing {self.__class__.__name__}",
        #     unit="step",
        #     bar_format="{l_bar}{bar:40}{r_bar}",
        #     colour='cyan'
        # ):
        self.best_routes, self.best_f = ant_colony_tsp(self.graph, self.pop_size, self.iterations, self.alpha, self.beta, self.rho, self.Q)
        # self.save(-1)
        # print(self.best_f)
        # plt.title(f"best fit {best_dist:.4f}")
        plt.grid()
        x = self.graph[self.best_routes, 0]
        x = np.append(x, x[0])
        y = self.graph[self.best_routes, 1]
        y = np.append(y, y[0])
        plt.plot(x, y, 'm')
        plt.plot(x, y, 'r.', markersize=15)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.show()
        return self.best_f, self.best_routes
        # return self.postrun


if __name__ == "__main__":
    pop_size = 10
    graph = Circle(pop_size)[:, :2]
    ar = AntRoute(pop_size, 100, 0.5,  0.5,  0.5, 100, graph).run(show_plot_animation=False, every=1)  # , graph
