import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


if __name__ == "__main__":
    from AlgoritmRoute import AlgoritmRoute
    from AlgoritmRoute import tqdm
    # from Circle import Circle
else:
    from .AlgoritmRoute import AlgoritmRoute
    from .AlgoritmRoute import tqdm
    # from .Circle import Circle


class GeneticRoute(AlgoritmRoute):
    def __init__(self, pop_size: int, iterations: int, child_size: int, mutation_probability: float, graph: np.ndarray | dict = None, **kwargs):
        super().__init__(pop_size, iterations, graph, **kwargs)
        self.child_size = child_size
        self.mutation_probability = mutation_probability

        self.f = np.inf * np.ones(self.pop_size + self.child_size)
        self.routes = np.zeros((self.pop_size + self.child_size, self.dim))
        for route in range(self.pop_size):
            self.routes[route, :] = np.arange(0, self.dim, 1)
            np.random.shuffle(self.routes[route, :])
            self.f[route] = self.fitness(self.routes[route, :], self.graph)
        self.best_f = np.min(self.f)  # best fitness in start population
        self.best_routes = self.routes[np.argmin(self.f), :].copy().astype(int)  # best routes in start population

        # print(self.graph)

    def run(self, **kwargs):
        self.prerun(**kwargs)
        for iteration in tqdm(
            range(self.iterations),
            desc="Processing",
            unit="step",
            bar_format="{l_bar}{bar:40}{r_bar}",
            colour='cyan'
        ):
            for child_route in range(self.pop_size, self.pop_size + self.child_size):
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
                self.routes[child_route, :] = child
                self.f[child_route] = self.fitness(child, self.graph)
            new_f = self.f[self.pop_size:]
            if self.best_f > np.min(new_f):
                self.best_f = np.min(new_f)
                self.best_routes = self.routes[np.argmin(new_f), :].copy().astype(int)
            self.routes = self.routes[np.lexsort(self.f[:, np.newaxis].T)]  # SORT BY FITNESS

            self.save

        return self.postrun

    # Fitness
    def fitness(self, routes, cities):
        routes_prev = routes.astype(int)
        routes_next = np.append(routes[1:], routes[0]).astype(int)
        f = (cities[routes_next, :] - cities[routes_prev, :])**2
        return np.sum(np.sqrt(f[:, 0] + f[:, 1]))

    # Crossover
    def __crossover(self, routes1, routes2):
        cross = np.random.randint(1, len(routes1))
        C1 = np.concatenate((routes1[:cross], routes2[cross:], routes2[:cross], routes1[cross:]))
        C1 = C1[sorted(np.unique(C1, return_index=True)[1])]
        return C1

    # Mutation
    def __mutation(self, routes):
        n = len(routes)
        i = np.random.randint(n)
        j = np.random.randint(n)
        while i == j:
            j = np.random.randint(n)
        if i > j:
            i, j = j, i
        if np.random.rand() < 0.5:
            routes[i], routes[j] = routes[j], routes[i]
        else:
            routes[i:j] = routes[i:j][::-1]


if __name__ == "__main__":
    GeneticRoute(200, 3000, 1000, 0.5).run(show_animation=True, show_convergence=False, every=100)

# N = 10
# P1 = np.arange(0, N, 1)
# np.random.shuffle(P1)
# # print(P1)
# P2 = np.arange(0, N, 1)
# np.random.shuffle(P2)

# # Crossover
# cross = np.random.randint(1, N)
# C1 = np.concatenate((P1[:cross], P2[cross:], P2[:cross], P1[cross:]))
# # print(C1)
# # print()
# C1 = C1[sorted(np.unique(C1, return_index=True)[1])]
# # print(P1)
# # print(P2)
# # print(cross)
# # print(C1)

# # Mutation
# cros1 = np.random.randint(N)
# cros2 = np.random.randint(N)

# while cros1 == cros2:
#     cros2 = np.random.randint(N)

# if cros1 > cros2:
#     cros1, cros2 = cros2, cros1

# # print(C1)
# if np.random.rand() < 0.5:
#     C1[cros1], C1[cros2] = C1[cros2], C1[cros1]
# else:
#     C1[cros1:cros2] = C1[cros1:cros2][::-1]
# # print(C1)

# # Population
# pop_size = 20
# parts = np.zeros((pop_size, N), dtype=int)
# for i in range(pop_size):
#     parts[i] = np.arange(0, N, 1)
#     np.random.shuffle(parts[i])
# print(parts)

# def plot(cities, Best_routes):
#     x = cities[Best_routes, 0]
#     y = cities[Best_routes, 1]
#     X = np.append(x, x[0])
#     Y = np.append(y, y[0])
#     plt.plot(X, Y, 'm')
#     plt.plot(X, Y, 'r.')

# # Fitness
# def fitness(routes, cities):
#     routes_prev = routes.astype(int)
#     # print(routes_prev)
#     # print(routes[1:], routes[0])
#     routes_next = np.append(routes[1:], routes[0]).astype(int)
#     f = (cities[routes_next, :] - cities[routes_prev, :])**2
#     return np.sum(np.sqrt(f[:, 0] + f[:, 1]))

# # Crossover
# def crossover(routes1, routes2):
#     cross = np.random.randint(1, len(routes1))
#     C1 = np.concatenate((routes1[:cross], routes2[cross:], routes2[:cross], routes1[cross:]))
#     C1 = C1[sorted(np.unique(C1, return_index=True)[1])]
#     return C1


# # Mutation
# def mutation(routes):
#     n = len(routes)
#     i = np.random.randint(n)
#     j = np.random.randint(n)
#     while i == j:
#         j = np.random.randint(n)
#     if i > j:
#         i, j = j, i
#     if np.random.rand() < 0.5:
#         routes[i], routes[j] = routes[j], routes[i]
#     else:
#         routes[i:j] = routes[i:j][::-1]

# N = 10
# N_Pop = 100
# mutation_probability = 0.7
# df = 2 * np.pi / N
# Max_iter = 300

# dfi = 2 * np.pi / N

# cities = np.zeros((N, 2))
# routes = np.zeros((N, N_Pop))
# f = np.inf * np.ones(N + N_Pop)
# for i in range(N):
#     cities[i, 0] = np.cos(dfi * i)
#     cities[i, 1] = np.sin(dfi * i)
# #     plt.plot(cities[:i+1, 0], cities[:i+1, 1], 'ro')
# #     plt.pause(0.01)
# # plt.show()

# best_f = float('inf')
# best_routes = None
# for i in range(N_Pop):
#     routes[:, i] = np.arange(0, N, 1)
#     np.random.shuffle(routes[:, i])
#     f[i] = fitness(routes[:, i], cities)
# #     if best_f > f[i]:
# #         best_f = f[i]
# #         best_routes = routes[:, i].copy()
# #     print(i, best_f)
# #     plt.plot(i, f[i], 'ro', i, best_f, 'go')
# #     plt.pause(0.1)
# # plt.show()
# Len = []
# best_f = np.min(f)  # best fitness in start population
# best_routes = routes[:, np.argmin(f)].copy()  # best routes in start population
# # print(best_f, best_routes)
# for k in range(Max_iter):
#     routes_children = np.zeros((N, N_Pop))
#     for L in range(N_Pop):
#         i = np.random.randint(N_Pop)  # Select parent 1
#         j = np.random.randint(N_Pop)  # Select parent 2
#         while i == j:
#             j = np.random.randint(N_Pop)
#         if i > j:
#             i, j = j, i
#         if np.random.rand() < 0.5:  # Crossover
#             child = crossover(routes[:, i], routes[:, j])
#         else:
#             child = crossover(routes[:, j], routes[:, i])
#         if np.random.rand() < mutation_probability:
#             mutation(child)  # Mutation
#         routes_children[:, L] = child
#     data = np.append(routes, routes_children, axis=0)  # Append parents and children
#     # print(routes.shape[0], data.shape[0])
#     for j in range(routes.shape[0], data.shape[0]):
#         f[j] = fitness(data[j, :], cities)  # Calculate fitness for children
#         # print(j, f[j])
#     if best_f > np.min(f[routes.shape[0]:]):
#         best_f = np.min(f[routes.shape[0]:])  # best fitness in start population
#         best_routes = routes[:, np.argmin(f[routes.shape[0]:])].copy().astype(int)  # best routes in start population
#     # print(data)
#     # print(f[:, np.newaxis].T)
#     # print(f[:, np.newaxis])
#     print(data, data.shape)
#     print(f[:, np.newaxis].T, f[:, np.newaxis].T.shape)
#     data = data[np.lexsort(f[:, np.newaxis].T)]  # SORT BY FITNESS
#     data = data[:N_Pop]  # Select top N_Pop

#     Len.append(best_f)

#     # if k % 2 == 0:
#     plt.cla()
#     plot(cities, best_routes)
#     plt.pause(0.1)
#     print("Generation: ", k, "Best routes: ", best_routes, "Best fit: ", best_f)
# plt.show()
# plt.plot(Len)
# # Opt_len = fitness(best_routes, cities)
# # print(f)

