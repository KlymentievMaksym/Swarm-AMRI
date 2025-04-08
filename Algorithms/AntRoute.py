import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


import threading
import numba as nb
import concurrent.futures
from multiprocessing import Pool

if __name__ == "__main__" or __name__ == "__mp_main__":
    from AlgoritmRoute import AlgoritmRoute
    from AlgoritmRoute import tqdm
    # from Circle import Circle
else:
    from .AlgoritmRoute import AlgoritmRoute
    from .AlgoritmRoute import tqdm
    # from .Circle import Circle


class AntRoute(AlgoritmRoute):
    def __init__(self, pop_size: int, iterations: int, alpha: float, beta: float, rho: float, Q: int, graph: np.ndarray | dict = None, threads_count: int = 4, **kwargs):
        super().__init__(pop_size, iterations, graph, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        # self.Q = np.random.randint(1, 111)
        self.Q = Q

        self.routes = np.zeros((self.pop_size, self.dim), dtype=int)
        self.routes[:, 0] = np.random.randint(0, self.dim, self.pop_size)
        self.pheromon_level = np.zeros((self.dim, self.dim)) + 1e-6
        self.parts_taboo_list = np.zeros((self.pop_size, self.dim), dtype=int)
        # for route_index in range(self.pop_size):
        #     self.parts_taboo_list[route_index, self.routes[route_index, 0]] = 1

        self.next_index = np.ones(self.pop_size, dtype=int)
        # print(self.routes)
        # print(self.parts_taboo_list)

        self.history_pheromon_level = []
        self.history_parts_taboo_list = []

        self.threads_count = threads_count

    def __forget(self, route_index: int):
        self.parts_taboo_list[route_index] = np.zeros((self.dim), dtype=int)
        self.parts_taboo_list[route_index, self.routes[route_index, 0]] = 1
        self.next_index[route_index] = 1

    def __choose_edge(self, route_index: int):
        available_edges = np.where(self.parts_taboo_list[route_index] == 0)[0]
        # print(available_edges)
        probability = np.zeros_like(available_edges, dtype=float)
        summ = 0
        for edge in available_edges:
            summ += self.pheromon_level[self.routes[route_index, self.next_index[route_index]-1], edge] ** self.alpha * 1/self.distance(self.graph[self.routes[route_index, self.next_index[route_index]-1], :], self.graph[edge, :]) ** self.beta
        for i, edge in enumerate(available_edges):
            probability[i] = (self.pheromon_level[self.routes[route_index, self.next_index[route_index]-1], edge] ** self.alpha * 1/self.distance(self.graph[self.routes[route_index, self.next_index[route_index]-1], :], self.graph[edge, :]) ** self.beta)/summ
        # print(probability)
        # print(sum(probability))
        self.routes[route_index, self.next_index[route_index]] = np.random.choice(available_edges, p=probability)
        self.parts_taboo_list[route_index, self.routes[route_index, self.next_index[route_index]]] = 1

    def __pheromon_update(self, route_index: int):
        # TODO update pheromon for all routes
        pheromon_to_add = self.Q/self.fitness(self.routes[route_index, :self.next_index[route_index] + 1], self.graph)

        self.delta_pheromon[self.routes[route_index, self.next_index[route_index]-1], self.routes[route_index, self.next_index[route_index]]] += pheromon_to_add
        self.delta_pheromon[self.routes[route_index, self.next_index[route_index]], self.routes[route_index, self.next_index[route_index]-1]] += pheromon_to_add
        # self.pheromon_level[self.routes[route_index, self.next_index[route_index]-1], self.routes[route_index, self.next_index[route_index]]] = (1 - self.rho) * self.pheromon_level[self.routes[route_index, self.next_index[route_index]-1], self.routes[route_index, self.next_index[route_index]]] + self.Q/self.fitness(self.routes[route_index, :self.next_index[route_index] + 1], self.graph)
        self.next_index[route_index] += 1

    def run(self, **kwargs):
        self.lock = threading.Lock()

        self.prerun(**kwargs)
        for iteration in tqdm(
            range(self.iterations),
            desc=f"Processing {self.__class__.__name__}",
            unit="step",
            bar_format="{l_bar}{bar:40}{r_bar}",
            colour='cyan'
        ):
            self.delta_pheromon = np.zeros_like(self.pheromon_level)

            # threads = []
            # size = self.pop_size / self.threads_count
            # for route_index in range(self.pop_size):
            #     thread = threading.Thread(target=self.ant_way, args=(route_index,))
            # for thread_index in range(self.threads_count):
            #     thread = threading.Thread(target=self.ant_way, args=(np.arange(self.pop_size + thread_index * size, self.pop_size + (thread_index + 1) * size, 1, dtype=int),))
            #     threads.append(thread)
            #     thread.start()

            # for thread in threads:
            #     thread.join()

            # pool = concurrent.futures.ThreadPoolExecutor(self.pop_size)
            # for route_index in range(self.pop_size):
            #     pool.submit(self.ant_way, route_index)
            # pool.shutdown(wait=True)
            
            # with Pool() as pool:
            #     pool.map(self.ant_way, self.pop_size)
            # pool2 = Pool(self.pop_size).map(self.ant_way, range(self.pop_size))
            # pool2.
            self.ant_way(np.arange(self.pop_size))
            self.pheromon_level = (1 - self.rho) * self.pheromon_level + self.delta_pheromon
            self.save(iteration)

        print(self.fitness(self.best_routes, self.graph))
        # print(self.pheromon_level)
        return self.postrun

    # def ant_way(self, route_index: int):
    def ant_way(self, route_indexes: np.ndarray):
        for route_index in route_indexes:
            self.__forget(route_index)
            for _ in range(1, self.dim):
                self.__choose_edge(route_index)
                self.__pheromon_update(route_index)
            fitness = self.fitness(self.routes[route_index, :], self.graph)
            if self.best_f > fitness:
                self.best_f = fitness
                self.best_routes = self.routes[route_index, :].astype(int)

    # def save(self, iteration: int):
    #     super().save(iteration)
    #     self.history_pheromon_level.append(self.pheromon_level.copy())
    #     self.history_parts_taboo_list.append(self.parts_taboo_list.copy())


if __name__ == "__main__":
    # graph = {0: [(1, 10), (3, 5)], 1: [(0, 10), (2, 5)], 2: [(1, 5), (3, 10)], 3: [(0, 5), (2, 10)]}
    ar = AntRoute(10, 100, 0.5,  0.5,  0.5, 100).run(show_plot_animation=True, every=1)  # , graph

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import networkx as nx

# class AntRoute:
#     def __init__(self, pop_size: int, iterations: int, alpha: float, beta: float, rho: float, graph: np.ndarray | dict):
#         self.pop_size = pop_size
#         self.iterations = iterations
#         self.alpha = alpha
#         self.beta = beta
#         self.rho = rho
#         self.Q = 100

#         # Handle dict or matrix graph input
#         if isinstance(graph, dict):
#             self.names = list(graph.keys())
#             self.dim = len(self.names)
#             matrix = np.zeros((self.dim, self.dim))
#             for i, key in enumerate(self.names):
#                 for neighbor, weight in graph[key]:
#                     j = self.names.index(neighbor)
#                     matrix[i, j] = weight
#             self.graph = matrix
#         else:
#             if graph.shape[0] != graph.shape[1]:
#                 raise ValueError("Graph must be square")
#             self.graph = graph
#             self.dim = graph.shape[0]
#             self.names = list(range(self.dim))

#         # Pheromone initialization
#         self.pheromon_level = np.ones((self.dim, self.dim)) * 1e-6

#         # History for animation
#         self.history_routes = []
#         self.best_path = None
#         self.best_cost = np.inf

#     def _route_cost(self, path):
#         return sum(self.graph[path[i], path[i+1]] for i in range(len(path)-1))

#     def _choose_next_city(self, current, visited):
#         probs = []
#         for j in range(self.dim):
#             if j in visited or self.graph[current][j] == 0:
#                 probs.append(0)
#             else:
#                 tau = self.pheromon_level[current][j] ** self.alpha
#                 eta = (1.0 / self.graph[current][j]) ** self.beta
#                 probs.append(tau * eta)
#         total = sum(probs)
#         if total == 0:
#             return np.random.choice([j for j in range(self.dim) if j not in visited])
#         probs = [p / total for p in probs]
#         return np.random.choice(range(self.dim), p=probs)

#     def _construct_solution(self):
#         all_routes = []
#         for _ in range(self.pop_size):
#             path = [np.random.randint(self.dim)]
#             while len(path) < self.dim:
#                 next_city = self._choose_next_city(path[-1], path)
#                 path.append(next_city)
#             path.append(path[0])  # complete the tour
#             all_routes.append(path)
#         return all_routes

#     def _update_pheromones(self, routes):
#         self.pheromon_level *= (1 - self.rho)
#         for route in routes:
#             cost = self._route_cost(route)
#             for i in range(len(route)-1):
#                 a, b = route[i], route[i+1]
#                 self.pheromon_level[a][b] += self.Q / cost
#                 self.pheromon_level[b][a] += self.Q / cost

#     def run(self):
#         for it in range(self.iterations):
#             routes = self._construct_solution()
#             self._update_pheromones(routes)

#             # Save best
#             for route in routes:
#                 cost = self._route_cost(route)
#                 if cost < self.best_cost:
#                     self.best_cost = cost
#                     self.best_path = route

#             self.history_routes.append(routes)
#             print(f"Iteration {it}: Best cost = {self.best_cost}")
#         self.plot()


#     def plot(self):
#         fig, ax = plt.subplots()
#         G = nx.Graph()

#         # Build the graph
#         for i in range(self.dim):
#             for j in range(self.dim):
#                 if self.graph[i][j] > 0:
#                     G.add_edge(i, j, weight=self.graph[i][j])

#         pos = nx.spring_layout(G, seed=42)  # consistent layout

#         def update(frame):
#             ax.clear()
#             ax.set_title(f"Iteration {frame}")
            
#             # Draw base graph
#             nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray')
#             nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{self.graph[i][j]:.0f}" for i, j in G.edges()}, ax=ax)

#             # Plot all ant paths in this iteration
#             for route in self.history_routes[frame]:
#                 path_edges = list(zip(route[:-1], route[1:]))
#                 nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=1, alpha=0.4, ax=ax)

#             # Plot best path so far
#             if self.best_path:
#                 best_edges = list(zip(self.best_path[:-1], self.best_path[1:]))
#                 nx.draw_networkx_edges(G, pos, edgelist=best_edges, edge_color='red', width=2, ax=ax)

#         ani = animation.FuncAnimation(fig, update, frames=len(self.history_routes), interval=1000, repeat=False)
#         plt.show()



# if __name__ == "__main__":
#     # Graph as adjacency list
#     graph = {
#         0: [(1, 2), (2, 9), (3, 4)],
#         1: [(0, 2), (2, 6), (3, 3)],
#         2: [(0, 9), (1, 6), (3, 8)],
#         3: [(0, 4), (1, 3), (2, 8)],
#     }
#     aco = AntRoute(pop_size=10, iterations=20, alpha=1, beta=2, rho=0.1, graph=graph)
#     aco.run()
