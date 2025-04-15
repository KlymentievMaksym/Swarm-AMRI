import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

import numba as nb

# print(__name__)
if __name__ == "__main__" or __name__ == "AlgoritmRoute":
    from Circle import Circle
else:
    from ..TSP.Circle import Circle


class AlgoritmRoute:
    def __init__(self, pop_size: int, iterations: int, graph: np.ndarray | None, **kwargs):
        self.pop_size = pop_size
        self.iterations = iterations

        available_types = {np.ndarray: "matrix", type(None): "None"}
        self.graph_type = available_types.get(type(graph), None)
        if self.graph_type is None:
            raise TypeError(f"Graph type {type(graph)} not supported")
        self.graph = graph

        # TODO REDO dict to matrix
        match self.graph_type:
            # case "dict":
            #     self.dim = len(self.graph)
            #     self.names = list(self.graph.keys())
            #     matrix = np.zeros((self.dim, self.dim))
            #     for key, values in self.graph.items():
            #         for val in values:
            #             matrix[self.names.index(key), self.names.index(val[0])] = val[1]
            #     self.graph = matrix
            case "matrix":
                self.graph = graph
                self.dim = self.graph.shape[0]
            case "None":
                self.graph = Circle(self.pop_size)[:, :2]
                self.dim = self.graph.shape[0]

        self.dist = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                # self.dist[i][j] = self.distance(self.graph[i], self.graph[j])
                self.dist[i][j] = np.linalg.norm(self.graph[i] - self.graph[j])
        self.best_f = float('inf')
        self.best_routes = None

        self.history_best_f = np.zeros(self.iterations)
        self.history_best_routes = np.zeros((self.iterations, self.dim)).astype(int)

        self.kwargs = kwargs

    def distance(self, a: np.ndarray, b: np.ndarray):
        return np.sqrt(np.sum((b - a) ** 2))

    # @property
    def save(self, iteration: int):
        self.history_best_f[iteration] = self.best_f
        self.history_best_routes[iteration] = self.best_routes.copy()

    @property
    def define_kwargs(self):

        self.plot_do = self.kwargs.get("plot", True)

        self.show_every = self.kwargs.get("every", 1)
        self.interval = self.kwargs.get("interval", 30)
        self.fps = self.kwargs.get("fps", 30)

        self.plot_convergence = self.kwargs.get("plot_convergence", True)
        self.show_convergence = self.kwargs.get("show_convergence", True)
        self.show_convergence_animation = self.kwargs.get("show_convergence_animation", False)
        self.close_convergence = self.kwargs.get("close_convergence", not self.show_convergence)
        self.save_convergence = self.kwargs.get("save_convergence", False)

        self.plot_plot = self.kwargs.get("plot_plot", True)
        self.show_plot = self.kwargs.get("show_plot", True)
        self.show_plot_animation = self.kwargs.get("show_plot_animation", False)
        self.close_plot = self.kwargs.get("close_plot", not self.show_plot)
        self.save_plot = self.kwargs.get("save_plot", False)

    def prerun(self, **kwargs):
        self.kwargs.update(kwargs)
        self.define_kwargs

    def fitness(self, routes: np.ndarray, cities: np.ndarray):
        routes_prev = routes.astype(int)
        routes_next = np.append(routes_prev[1:], routes_prev[0])#.astype(int)

        f = (cities[routes_next, :] - cities[routes_prev, :])**2

        return np.sum(np.sqrt(f[:, 0] + f[:, 1]))

    @property
    def postrun(self):
        if self.plot_do:
            self.plot
        return self.best_f, self.best_routes

    @property
    def plot(self):
        if self.plot_convergence:
            if not self.show_convergence_animation:
                plt.title(f"best fit {self.history_best_f[-1]:.4f}")
                plt.plot(self.history_best_f)
                plt.grid()
                # plt.legend()
                plt.xlabel("Iterations")
                plt.ylabel("Fitness")

                if self.save_convergence:
                    try:
                        plt.savefig(self.save_convergence)
                    except Exception as e:
                        print(e)
            else:
                def update(frame):
                    plt.cla()
                    plt.title(f"iteration {frame * self.show_every}, best fit {self.history_best_f[frame * self.show_every]:.4f}")
                    plt.plot(self.history_best_f[:frame * self.show_every])
                    plt.grid()
                    # plt.legend()
                    plt.xlabel("Iterations")
                    plt.ylabel("Fitness")
                ani = animation.FuncAnimation(plt.gcf(), update, frames=self.iterations//self.show_every, interval=self.interval)

                if self.save_convergence:
                    try:
                        ani.save(self.save_convergence, fps=self.fps)
                    except Exception as e:
                        print(e)

            if self.show_convergence:
                plt.show()
            elif self.close_convergence:
                plt.close()

        if self.plot_plot:
            if not self.show_plot_animation:
                # plt.cla()
                plt.title(f"best fit {self.history_best_f[-1]:.4f}")
                plt.grid()
                x = self.graph[self.history_best_routes[-1], 0]
                x = np.append(x, x[0])
                y = self.graph[self.history_best_routes[-1], 1]
                y = np.append(y, y[0])
                plt.plot(x, y, 'm')
                plt.plot(x, y, 'r.', markersize=15)
                plt.xlabel("x")
                plt.ylabel("y")

                if self.save_plot:
                    try:
                        plt.savefig(self.save_plot)
                    except Exception as e:
                        print(e)
            else:
                def update(frame):
                    plt.cla()
                    plt.title(f"iteration {frame * self.show_every}, best fit {self.history_best_f[frame * self.show_every]:.4f}")
                    x = self.graph[self.history_best_routes[frame * self.show_every], 0]
                    x = np.append(x, x[0])
                    y = self.graph[self.history_best_routes[frame * self.show_every], 1]
                    y = np.append(y, y[0])
                    plt.plot(x, y, 'm')
                    plt.plot(x, y, 'r.', markersize=15)
                    plt.xlabel("x")
                    plt.ylabel("y")
                    # plt.plot(self.history_best_f[frame])
                ani = animation.FuncAnimation(plt.gcf(), update, frames=self.iterations//self.show_every, interval=self.interval)

                if self.save_plot:
                    try:
                        ani.save(self.save_plot, fps=self.fps)
                    except Exception as e:
                        print(e)

            if self.show_plot:
                plt.show()
            elif self.close_plot:
                plt.close()
