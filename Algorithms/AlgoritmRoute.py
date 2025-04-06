import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# if __name__ == "__main__":
from Circle import Circle
# else:
#     from .Circle import Circle


class AlgoritmRoute:
    def __init__(self, pop_size: int, iterations: int, graph: np.ndarray | dict, **kwargs):
        self.pop_size = pop_size
        self.iterations = iterations

        available_types = {dict: "dict", np.ndarray: "matrix", type(None): "None"}
        self.graph_type = available_types.get(type(graph), None)
        if self.graph_type is None:
            raise TypeError(f"Graph type {type(graph)} not supported")
        self.graph = graph

        # TODO REDO dict to matrix
        match self.graph_type:
            case "dict":
                self.dim = len(self.graph)
                self.names = list(self.graph.keys())
                matrix = np.zeros((self.dim, self.dim))
                for key, values in self.graph.items():
                    for val in values:
                        matrix[self.names.index(key), self.names.index(val[0])] = val[1]
                self.graph = matrix
            case "matrix":
                if self.graph.shape[0] != self.graph.shape[1]:
                    raise ValueError("Graph must be a square matrix")
                self.dim = self.graph.shape[0]
            case "None":
                self.graph = Circle(self.pop_size)[:, :2]
                self.dim = self.graph.shape[0]

        self.best_f = float('inf')
        self.best_routes = None

        self.history_best_f = []
        self.history_best_routes = []

        self.kwargs = kwargs

    @property
    def save(self):
        self.history_best_f.append(self.best_f)
        self.history_best_routes.append(self.best_routes.copy())

    @property
    def define_kwargs(self):

        self.plot_do = self.kwargs.get("plot", True)

        self.show_every = self.kwargs.get("every", 1)

        self.plot_convergence = self.kwargs.get("plot_convergence", True)
        self.show_convergence = self.kwargs.get("show_convergence", True)
        self.close_convergence = self.kwargs.get("close_convergence", not self.show_convergence)

        self.plot_animation = self.kwargs.get("plot_animation", True)
        self.show_animation = self.kwargs.get("show_animation", False)
        # self.close_animation = self.kwargs.get("close_animation", not self.show_animation)

    def prerun(self, **kwargs):
        self.kwargs.update(kwargs)
        self.define_kwargs

    @property
    def postrun(self):
        if self.plot_do:
            self.plot
        return self.best_f, self.best_routes

    @property
    def plot(self):
        if self.plot_convergence:
            plt.plot(self.history_best_f, label=f"Best fit {self.history_best_f[-1]:.4f}")
            plt.grid()
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Fitness")
            if self.show_convergence:
                plt.show()
            elif self.close_convergence:
                plt.close()

        if self.plot_animation:
            def update(frame):
                plt.cla()
                plt.title(f"iteration {frame * self.show_every}, best fit {self.history_best_f[frame * self.show_every]:.4f}")
                x = self.graph[self.history_best_routes[frame * self.show_every], 0]
                x = np.append(x, x[0])
                y = self.graph[self.history_best_routes[frame * self.show_every], 1]
                y = np.append(y, y[0])
                plt.plot(x, y, 'm')
                plt.plot(x, y, 'r.', markersize=15)
                # plt.plot(self.history_best_f[frame])
            ani = animation.FuncAnimation(plt.gcf(), update, frames=self.iterations//self.show_every, interval=100)
            if self.show_animation:
                plt.show()
