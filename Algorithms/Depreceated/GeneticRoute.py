import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import threading
import numba as nb
import concurrent.futures
from multiprocessing import Pool


# print(__name__)
if __name__ == "__main__" or __name__ == "__mp_main__":
    from Algorithms.Depreceated.AlgoritmRoute import AlgoritmRoute
    from Algorithms.Depreceated.AlgoritmRoute import tqdm
    # from Circle import Circle
else:
    from .AlgoritmRoute import AlgoritmRoute
    from .AlgoritmRoute import tqdm
    # from .Circle import Circle


class GeneticRoute(AlgoritmRoute):
    def __init__(self, pop_size: int, iterations: int, child_size: int, mutation_probability: float, graph: np.ndarray | dict = None, threads_count: int = 1, **kwargs):
        super().__init__(pop_size, iterations, graph, **kwargs)
        self.child_size = child_size
        self.mutation_probability = mutation_probability

        self.f = np.inf * np.ones(self.pop_size + self.child_size)
        self.routes = np.zeros((self.pop_size + self.child_size, self.dim))
        for route in range(self.pop_size):
            self.routes[route, :] = np.arange(0, self.dim, 1)
            np.random.shuffle(self.routes[route, :])
            self.f[route] = self.fitness(self.routes[route, :], self.graph)


        self.threads_count = threads_count
        # self.lock = threading.Lock()  # for best_f and best_routes access

    def run(self, **kwargs):
        self.prerun(**kwargs)
        for iteration in tqdm(
            range(self.iterations),
            desc=f"Processing {self.__class__.__name__}",
            unit="step",
            bar_format="{l_bar}{bar:40}{r_bar}",
            colour='cyan'
        ):
            # pool = Pool(self.threads_count)
            # values = pool.map(self.__make_children, range(self.pop_size, self.pop_size + self.child_size))
            # print(values)
            # child_size = self.child_size/self.threads_count
            # threads = []

            # for thread_index in range(self.threads_count):
            #     x = threading.Thread(target=self.__make_children, args=(np.arange(self.pop_size + thread_index * child_size, self.pop_size + (thread_index + 1) * child_size, 1, dtype=int), ))
            #     threads.append(x)
            #     x.start()
            # for thread in threads:
            #     thread.join()
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     self.f = np.array(list(executor.map(self.fitness, self.routes, [self.graph for _ in range(self.routes.shape[0])])))
            # print(self.f.shape)
            # pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.threads_count)
            # for thread_index in range(self.threads_count):
            #     pool.submit(self.__make_children, np.arange(self.pop_size + thread_index * child_size, self.pop_size + (thread_index + 1) * child_size, 1, dtype=int))
            # pool.shutdown(wait=True)

            self.__make_children(np.arange(self.pop_size, self.pop_size + self.child_size, 1, dtype=int))
            argsort = np.argsort(self.f)
            self.routes = self.routes[argsort]  # SORT BY FITNESS
            self.f = self.f[argsort]
            if self.best_f > self.f[0]:
                self.best_f = self.f[0]
                self.best_routes = self.routes[0, :].astype(int)

            self.save(iteration)
        print(self.fitness(self.best_routes, self.graph))
        return self.postrun

    def __make_children(self, child_routes: np.ndarray):
        for child_route in child_routes:
            i = np.random.randint(self.pop_size)  # Select parent 1
            j = np.random.randint(self.pop_size)  # Select parent 2
            while i == j:
                j = np.random.randint(self.pop_size)
            if i > j:
                i, j = j, i
            if np.random.rand() < 0.5:  # Crossover
                child = self.__crossover(self.routes[i, :], self.routes[j, :])
            else:
                child = self.__crossover(self.routes[j, :], self.routes[i, :])
            if np.random.rand() < self.mutation_probability:
                self.__mutation(child)  # Mutation
            self.routes[child_route, :] = child.copy()
            self.f[child_route] = self.fitness(child, self.graph)

    def __crossover(self, routes1: np.ndarray, routes2: np.ndarray):
        cross = np.random.randint(1, len(routes1))
        C1 = np.concatenate((routes1[:cross], routes2[cross:], routes2[:cross], routes1[cross:]))
        C1 = C1[sorted(np.unique(C1, return_index=True)[1])]
        return C1

    def __mutation(self, route: np.ndarray):
        n = len(route)
        i = np.random.randint(n)
        j = np.random.randint(n)
        while i == j:
            j = np.random.randint(n)
        if i > j:
            i, j = j, i
        if np.random.rand() < 0.5:
            route[i], route[j] = route[j], route[i]
        else:
            route[i:j] = route[i:j][::-1]


if __name__ == "__main__":
    # GeneticRoute(300, 1000, 1000, 0.5).run(show_animation=True, show_convergence=False, every=1)
    GeneticRoute(20, 100, 100, 0.5).run(show_plot_animation=True, show_convergence=True, show_convergence_animation=False, plot_convergence=True, plot_plot=True, every=1) #, save_animation="./Lab4/Images/an.2.gif", save_convergence="./Lab4/Images/conv.2.gif")
