import numpy as np
from os import listdir

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

import threading
import numba as nb
import concurrent.futures
from multiprocessing import Pool

from Algorithms.Circle import Circle

datas = []
for path in listdir("./Lab4/Data"):
    with open(f"./Lab4/Data/{path}", 'r') as fr:
        text = fr.read()

    info_data = text.split("NODE_COORD_SECTION")

    info = info_data[0]
    info = info.split("\n")
    info = info[:-1]
    info = {part.split(" ")[0]: part.split(" ")[2] for part in info}

    data = info_data[1]
    data = data.split("\n")
    data = data[1:-2]
    data = np.array([list(map(int, part.split(" "))) for part in data])
    datas.append(data)


@nb.njit()
def fitness(routes: np.ndarray, cities: np.ndarray):
    routes_prev = routes
    routes_next = np.append(routes_prev[1:], routes_prev[0])#.astype(int)

    f = (cities[routes_next, :] - cities[routes_prev, :])**2

    return np.sum(np.sqrt(f[:, 0] + f[:, 1]))


@nb.njit()
def distance(a: np.ndarray, b: np.ndarray):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


# @nb.njit()
# def calculate_sum(next_index, edges, summ, route_index, pheromon_level, routes, graph, alpha, beta):
#     next_indexx = next_index[route_index]
#     for edge in edges:
#         summ[route_index] += pheromon_level[routes[route_index, next_indexx-1], edge] ** alpha * 1/distance(graph[routes[route_index, next_indexx-1], :], graph[edge, :]) ** beta


# # @nb.njit()
# def calculate_probability(next_index, edges, summ, route_index, pheromon_level, routes, graph, alpha, beta):
#     next_indexx = next_index[route_index]
#     for prob_index, edge in enumerate(edges):
#         probability[route_index, prob_index] = (pheromon_level[routes[route_index, next_indexx-1], edge] ** alpha * 1/distance(graph[routes[route_index, next_indexx-1], :], graph[edge, :]) ** beta)/summ[route_index]


# # @nb.njit()
# def choose_road(next_index, route_index, routes, available_edges, probability, parts_taboo_list):
#     next_indexx = next_index[route_index]
#     routes[route_index, next_indexx] = np.random.choice(available_edges[route_index], p=probability[route_index])
#     parts_taboo_list[route_index, routes[route_index, next_indexx]] = 1


# @nb.njit()
# def pheromon_update(routes, route_index, next_index, graph, delta_pheromon, pheromon_level):
    # pheromon_to_add = Q/fitness(routes[route_index, :next_index[route_index] + 1], graph)
    # edge_from, edge_to = routes[route_index, next_index[route_index] - 1], routes[route_index, next_index[route_index]]

    # delta_pheromon[edge_from, edge_to] += pheromon_to_add
    # delta_pheromon[edge_to, edge_from] += pheromon_to_add

    # pheromon_level[edge_from, edge_to] += pheromon_to_add
    # pheromon_level[edge_to, edge_from] += pheromon_to_add

    # next_index[route_index] += 1


pop_size = 20
iterations = 100
alpha = 0.5
beta = 0.5
rho = 0.5
Q = 100
threads_count = 4

graph = Circle(pop_size)[:, :2]
# graph = datas[0][:, 1:]
dim = graph.shape[0]

best_f = float('inf')
best_routes = None

routes = np.zeros((pop_size, dim), dtype=int)
integer = np.random.randint(0, dim)
routes[:, 0] = np.random.randint(0, dim, pop_size)
# routes[:, 0] = np.array(list(integer for i in range(pop_size)))
# print(routes)
pheromon_level = np.zeros((dim, dim)) + 1e-6
# parts_taboo_list = np.zeros((pop_size, dim), dtype=int)

history_best_f = np.zeros(iterations)
history_best_routes = np.zeros((iterations, dim)).astype(int)

# next_index = np.ones(pop_size, dtype=int)

pheromon_level_before = pheromon_level.copy()

for iteration in tqdm(
    range(iterations),
    desc="Processing",
    unit="step",
    bar_format="{l_bar}{bar:40}{r_bar}",
    colour='cyan'
):
# for iteration in range(iterations):
    delta_pheromon = np.zeros_like(pheromon_level)

    parts_taboo_list = np.zeros((pop_size, dim), dtype=int)
    for route_index in range(pop_size):
        parts_taboo_list[route_index, routes[route_index, 0]] = 1

    next_index = np.ones(pop_size, dtype=int)
    # print(parts_taboo_list.shape)
    for _ in range(1, dim):
        available_edges = np.array([np.where(parts_taboo_list[route_index] == 0)[0] for route_index in range(pop_size)])
        # print(available_edges.shape)
        probability = np.zeros_like(available_edges, dtype=float)
        summ = np.zeros(pop_size)

        for route_index, edges in enumerate(available_edges):
            # calculate_sum(next_index, edges, summ, route_index, pheromon_level, routes, graph, alpha, beta)
            next_indexx = next_index[route_index]
            for edge in edges:
                summ[route_index] += pheromon_level[routes[route_index, next_indexx-1], edge] ** alpha * 1/distance(graph[routes[route_index, next_indexx-1], :], graph[edge, :]) ** beta

            # calculate_probability(next_index, edges, summ, route_index, pheromon_level, routes, graph, alpha, beta)
            # next_indexx = next_index[route_index]
            for prob_index, edge in enumerate(edges):
                probability[route_index, prob_index] = (pheromon_level[routes[route_index, next_indexx-1], edge] ** alpha * 1/distance(graph[routes[route_index, next_indexx-1], :], graph[edge, :]) ** beta)/summ[route_index]
            # choose_road(next_index, route_index, routes, available_edges, probability, parts_taboo_list)
            # next_indexx = next_index[route_index]
            routes[route_index, next_indexx] = np.random.choice(available_edges[route_index], p=probability[route_index])
            parts_taboo_list[route_index, routes[route_index, next_indexx]] = 1

            # pheromon_update(routes, route_index, next_index, graph, delta_pheromon, pheromon_level)
            pheromon_to_add = Q/fitness(routes[route_index, :next_index[route_index] + 1], graph)
            edge_from, edge_to = routes[route_index, next_index[route_index] - 1], routes[route_index, next_index[route_index]]

            delta_pheromon[edge_from, edge_to] += pheromon_to_add
            delta_pheromon[edge_to, edge_from] += pheromon_to_add

            pheromon_level[edge_from, edge_to] += pheromon_to_add
            pheromon_level[edge_to, edge_from] += pheromon_to_add

            next_index[route_index] += 1
            # print(delta_pheromon)

    # print(routes)
    for route_index in range(pop_size):
        fitnes = fitness(routes[route_index, :], graph)
        if best_f > fitnes:
            best_f = fitnes
            best_routes = routes[route_index, :].copy()

        # print(np.sum(probability, axis=1))
        # print(summ)
        # if _ == 2:
        #     break
        # break
    # print(routes)
    # print(best_routes)
    pheromon_level = (1 - rho) * pheromon_level_before + delta_pheromon
    pheromon_level_before = pheromon_level.copy()
    # break
print(fitness(best_routes, graph))
plt.title(f"best fit {best_f:.4f}")
plt.grid()
# x = graph[history_best_routes[-1], 0]
x = graph[best_routes, 0]
x = np.append(x, x[0])
# y = graph[history_best_routes[-1], 1]
y = graph[best_routes, 1]
y = np.append(y, y[0])
plt.plot(x, y, 'm')
plt.plot(x, y, 'r.', markersize=15)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# for route_index in range(pop_size):
#     # parts_taboo_list[route_index] = np.zeros((dim), dtype=int)
#     # parts_taboo_list[route_index, routes[route_index, 0]] = 1
#     # next_index[route_index] = 1
#     for _ in range(1, dim):
#         available_edges = np.where(parts_taboo_list[route_index] == 0)[0]
#         probability = np.zeros_like(available_edges, dtype=float)
#         summ = 0
#         next_indexx = next_index[route_index]
#         for edge in available_edges:
#             summ += pheromon_level[routes[route_index, next_indexx-1], edge] ** alpha * 1/distance(graph[routes[route_index, next_indexx-1], :], graph[edge, :]) ** beta
#         for i, edge in enumerate(available_edges):
#             probability[i] = (pheromon_level[routes[route_index, next_indexx-1], edge] ** alpha * 1/distance(graph[routes[route_index, next_indexx-1], :], graph[edge, :]) ** beta)/summ
#         routes[route_index, next_indexx] = np.random.choice(available_edges, p=probability)
#         parts_taboo_list[route_index, routes[route_index, next_indexx]] = 1

#         pheromon_to_add = Q/fitness(routes[route_index, :next_indexx + 1], graph)
#         delta_pheromon[routes[route_index, next_indexx-1], routes[route_index, next_indexx]] += pheromon_to_add
#         delta_pheromon[routes[route_index, next_indexx], routes[route_index, next_indexx-1]] += pheromon_to_add
#         next_index[route_index] += 1

#     fitnes = fitness(routes[route_index, :], graph)
#     if best_f > fitnes:
#         best_f = fitnes
#         best_routes = routes[route_index, :].astype(int)
# pheromon_level = (1 - rho) * pheromon_level + delta_pheromon
# print(routes)
# print(pheromon_level)

# plt.clf()
# plt.imshow(pheromon_level, cmap='binary')
# plt.colorbar()
# plt.xticks(range(dim))
# plt.yticks(range(dim))
# plt.pause(0.01)

# history_best_f[iteration] = best_f
# history_best_routes[iteration] = best_routes.astype(int)

# print(fitness(best_routes, graph))
# plt.show()
















# threads = []
# size = pop_size / threads_count
# for route_index in range(pop_size):
#     thread = threading.Thread(target=ant_way, args=(route_index,))
# for thread_index in range(threads_count):
#     thread = threading.Thread(target=ant_way, args=(np.arange(pop_size + thread_index * size, pop_size + (thread_index + 1) * size, 1, dtype=int),))
#     threads.append(thread)
#     thread.start()

# for thread in threads:
#     thread.join()

# pool = concurrent.futures.ThreadPoolExecutor(pop_size)
# for route_index in range(pop_size):
#     pool.submit(ant_way, route_index)
# pool.shutdown(wait=True)

# with Pool() as pool:
#     pool.map(ant_way, pop_size)
# pool2 = Pool(pop_size).map(ant_way, range(pop_size))
# pool2.


# print(pheromon_level)