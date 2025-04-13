import numpy as np
from os import listdir
from time import time

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


# @nb.njit()
def fitness(routes: np.ndarray, cities: np.ndarray):
    routes_prev = routes
    routes_next = np.append(routes_prev[1:], routes_prev[0])

    f = (cities[routes_next, :] - cities[routes_prev, :])**2

    return np.sum(np.sqrt(f[:, 0] + f[:, 1]))


# @nb.njit()
def distance(a: np.ndarray, b: np.ndarray):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

start = time()
repeat_times = 10
bests = np.zeros(repeat_times)
# TODO PARALELISM
pop_size = 10
start_routes = np.zeros((pop_size, repeat_times), dtype=int)
for repeat in range(repeat_times):
    iterations = 100

    graph = Circle(pop_size)[:, :2]
    # graph = datas[0][:, 1:]

    dim = graph.shape[0]

    best_f = float('inf')
    best_routes = None

    alpha = 0.5
    beta = 0.5
    rho = 0.5

    Q = 100

    # routes[:, 0] = np.random.randint(0, dim, pop_size)
    # edge = np.random.randint(0, dim)
    # routes[:, 0] = np.array([edge] * pop_size)
    pheromon_level = np.zeros((dim, dim)) + 1e-6
    pheromon_level_before = pheromon_level.copy()
    # parts_taboo_list = np.zeros((pop_size, dim), dtype=int)

    # next_index = np.ones(pop_size, dtype=int)

    # history_pheromon_level = []
    # history_parts_taboo_list = []

    threads_count = 4

    for iteration in tqdm(
        range(iterations),
        desc="Processing",
        unit="step",
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour='cyan'
    ):
        delta_pheromon = np.zeros_like(pheromon_level)
        routes = np.zeros((pop_size, dim), dtype=int)
        routes[:, 0] = np.random.randint(0, dim, pop_size)
        next_index = np.ones(pop_size, dtype=int)
        parts_taboo_list = np.zeros((pop_size, dim), dtype=int)
        for route_index in range(pop_size):
            # RESET MEMORY
            parts_taboo_list[route_index, routes[route_index, 0]] = 1
            # DO Path
            for _ in range(1, dim):
                # Choose edge
                available_edges = np.where(parts_taboo_list[route_index] == 0)[0]
                probability = np.zeros_like(available_edges, dtype=float)
                summ = 0
                for edge in available_edges:
                    summ += pheromon_level[routes[route_index, next_index[route_index]-1], edge] ** alpha * 1/distance(graph[routes[route_index, next_index[route_index]-1], :], graph[edge, :]) ** beta
                for i, edge in enumerate(available_edges):
                    probability[i] = (pheromon_level[routes[route_index, next_index[route_index]-1], edge] ** alpha * 1/distance(graph[routes[route_index, next_index[route_index]-1], :], graph[edge, :]) ** beta)/summ
                routes[route_index, next_index[route_index]] = np.random.choice(available_edges, p=probability)
                parts_taboo_list[route_index, routes[route_index, next_index[route_index]]] = 1

                # Update pheromon
                pheromon_to_add = Q/fitness(routes[route_index, :next_index[route_index] + 1], graph)

                delta_pheromon[routes[route_index, next_index[route_index]-1], routes[route_index, next_index[route_index]]] += pheromon_to_add
                delta_pheromon[routes[route_index, next_index[route_index]], routes[route_index, next_index[route_index]-1]] += pheromon_to_add

                pheromon_level[routes[route_index, next_index[route_index]-1], routes[route_index, next_index[route_index]]] += pheromon_to_add
                pheromon_level[routes[route_index, next_index[route_index]], routes[route_index, next_index[route_index]-1]] += pheromon_to_add

                next_index[route_index] += 1
        # Calculate fitness for all routes at once
        fitness_values2 = np.apply_along_axis(lambda route: fitness(route, graph), 1, routes)
        best_index = np.argmin(fitness_values2)
        if best_f > fitness_values2[best_index]:
            best_f = fitness_values2[best_index]
            best_routes = routes[best_index, :].astype(int)

        # Pheromon decrease + update
        pheromon_level = (1 - rho) * pheromon_level_before + delta_pheromon
        pheromon_level_before = pheromon_level.copy()

    # RESULT
    result = fitness(best_routes, graph)
    # print(result)
    bests[repeat] = result
    start_routes[:, repeat] = routes[:, 0].copy()
print(np.mean(bests), np.max(bests), np.min(bests))
print(start_routes[:, np.argmax(bests)], start_routes[:, np.argmin(bests)])
print(time() - start, "seconds")
