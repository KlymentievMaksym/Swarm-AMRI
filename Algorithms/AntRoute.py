import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AntRoute:
    def __init__(self, pop_size: int, iterations: int, alpha: float, beta: float, rho: float, graph: np.ndarray | dict, **kwargs):
        self.pop_size = pop_size
        self.iterations = iterations

        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.Q = np.random.randint(1, 11)

        available_types = {dict: "dict", np.ndarray: "matrix"}
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
        # print(self.graph)
        self.parts = np.random.randint(0, self.dim, pop_size)
        self.parts = np.array([2, 3, 1, 2, 2, 0, 1, 2, 2, 2])
        self.parts_start = self.parts.copy()
        self.parts_best = np.zeros((self.pop_size, self.dim), dtype=int)
        self.parts_best[:, 0] = self.parts.copy()
        self.parts_best_count = np.ones(self.pop_size, dtype=int)
        self.best_parts = np.zeros(self.dim)
        self.best = float("inf")
        # print(self.parts)
        self.pheromon_level = np.zeros((self.dim, self.dim)) + 1e-6
        # print(self.pheromon_level)

        self.parts_taboo_list = np.zeros((self.pop_size, self.dim))
        # for part_index in range(self.pop_size):
        #     self.parts_taboo_list[part_index, self.parts[part_index]] = 1
        # print(self.parts_taboo_list)

        self.history_parts = []
        self.history_parts_best = []
        self.history_best = []
        self.history_best_parts = []
        # self.history_parts_prev = []
        self.history_pheromon_level = []
        self.history_parts_taboo_list = []

    def __forget(self, part_index: int):
        self.parts_taboo_list[part_index] = np.zeros(self.dim)
        # self.parts_taboo_list[part_index, part_index] = 1
        self.parts_taboo_list[part_index, self.parts_start[part_index]] = 1

    def __choose_edge(self, part_index: int):
        # available_edges = self.graph[part_index][self.graph[part_index] > 0]
        available_edges = np.where(np.logical_and((self.parts_taboo_list[part_index] == 0), (self.graph[self.parts[part_index]] > 0)))[0]
        # print(self.parts_taboo_list[part_index] == 0)
        # print(self.graph[part_index] > 0)
        # print(np.logical_and((self.parts_taboo_list[part_index] == 0), (self.graph[part_index] > 0)))
        # available_edges_summ = np.where((self.parts_taboo_list[part_index] == 0))[0]
        # available_edges_summ = np.where(np.logical_and((self.parts_taboo_list[part_index] == 0), (self.graph[part_index] > 0)))[0]
        probability = np.zeros_like(available_edges, dtype=float)
        summ = 0
        for edge in available_edges:
            summ += self.pheromon_level[self.parts[part_index], edge] ** self.alpha * 1/self.graph[self.parts[part_index], edge] ** self.beta
        for i, edge in enumerate(available_edges):
            probability[i] = (self.pheromon_level[self.parts[part_index], edge] ** self.alpha * 1/self.graph[self.parts[part_index], edge] ** self.beta)/summ
        # print(probability)
        # print(sum(probability))
        self.parts[part_index] = np.random.choice(available_edges, p=probability)
        self.parts_best[part_index, self.parts_best_count[part_index]] = self.parts[part_index].copy()
        self.parts_best_count[part_index] += 1

    def __pheromon_update(self, part_index: int):
        self.pheromon_level[self.parts_best[part_index, self.parts_best_count[part_index]], self.parts[part_index]] = (1 - self.rho) * self.pheromon_level[self.parts_best[part_index, self.parts_best_count[part_index]], self.parts[part_index]] + self.Q/self.graph[self.parts_best[part_index, self.parts_best_count[part_index]], self.parts[part_index]]
        # self.parts_taboo_list[part_index, self.parts_prev[part_index]] = 1

    def run(self, **kwargs):
        for iteration in range(self.iterations):
            for i in range(self.pop_size):
                self.__forget(i)
                # print(self.parts[i])
                self.__choose_edge(i)
                # print(self.parts[i])
                self.__pheromon_update(i)
                self.compare(i)
            self.save
            self.parts_best = np.zeros((self.pop_size, self.dim), dtype=int)
            self.parts_best[:, 0] = self.parts.copy()
            self.parts_best_count = np.ones(self.pop_size, dtype=int)
        #         raise NotImplementedError("Method not implemented")
        # raise NotImplementedError("Method not implemented")
        print(self.pheromon_level)
        self.plot

    def compare(self, part_index: int):
        if self.best < self.parts_best[part_index, self.parts_best_count[part_index]]:
            self.best = self.parts_best[part_index, self.parts_best_count[part_index]]
            self.best_parts = self.parts_best[part_index].copy()

    @property
    def save(self):
        self.history_parts.append(self.parts.copy())
        self.history_parts_best.append(self.parts_best.copy())
        self.history_best.append(self.best)
        self.history_best_parts.append(self.best_parts.copy())
        # self.history_parts_prev.append(self.parts_prev.copy())
        self.history_pheromon_level.append(self.pheromon_level.copy())
        self.history_parts_taboo_list.append(self.parts_taboo_list.copy())

    @property
    def plot(self):
        plt.imshow(self.graph>0, cmap="binary")
        plt.plot(self.history_best_parts[0], self.history_best[0])
        def update(frame):
            plt.cla()
            plt.title(f"iteration {frame}")
            plt.imshow(self.graph>0, cmap="binary")
            plt.plot(self.history_best_parts[frame], self.history_best[frame])
        ani = animation.FuncAnimation(plt.gcf(), update, frames=self.iterations, interval=1000)
        plt.show()


if __name__ == "__main__":
    # graph = np.array([
    #     [0, 1, 1, 0],
    #     [1, 0, 1, 0],
    #     [1, 1, 0, 1],
    #     [0, 0, 1, 0]
    # ])
    # ar = AntRoute(10, 10, .5, .5, .5, graph)
    graph = {0: [(1, 10), (3, 5)], 1: [(0, 10), (2, 5)], 2: [(1, 5), (3, 10)], 3: [(0, 5), (2, 10)]}
    # graph = {"A": [(1, 10), (3, 5)], 1: [("A", 10), (2, 5)], 2: [(1, 5), (3, 10)], 3: [("A", 5), (2, 10)]}
    ar = AntRoute(10, 10, .5, .5, .5, graph).run()
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
