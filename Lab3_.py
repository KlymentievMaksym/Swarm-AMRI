import numpy as np
import random
from Algorithms.Circle import Circle
from os import listdir
from tqdm import tqdm

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

def distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
    return dist

def ant_colony_tsp(cities, n_ants=10, n_iterations=100, alpha=1, beta=5, rho=0.5, Q=100):
    n = len(cities)
    dist = distance_matrix(cities)
    pheromone = np.ones((n, n)) / n
    best_distance = float("inf")
    best_tour = []

    for iteration in tqdm(
        range(n_iterations),
        desc="Processing",
        unit="step",
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour='cyan'
    ):
        all_tours = []
        all_distances = []

        for ant in range(n_ants):
            tour = [random.randint(0, n - 1)]
            while len(tour) < n:
                i = tour[-1]
                probs = []
                for j in range(n):
                    if j not in tour:
                        tau = pheromone[i][j] ** alpha
                        eta = (1 / dist[i][j]) ** beta
                        probs.append((j, tau * eta))
                total = sum(p[1] for p in probs)
                probs = [(city, p / total) for city, p in probs]
                r = random.random()
                cumulative = 0.0
                for city, p in probs:
                    cumulative += p
                    if r <= cumulative:
                        tour.append(city)
                        break

            tour_distance = sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))
            if tour_distance < best_distance:
                best_distance = tour_distance
                best_tour = tour
            all_tours.append(tour)
            all_distances.append(tour_distance)

        # Випаровування феромону
        pheromone *= (1 - rho)

        # Додавання нового феромону
        for tour, d in zip(all_tours, all_distances):
            for i in range(n):
                a = tour[i]
                b = tour[(i + 1) % n]
                pheromone[a][b] += Q / d
                pheromone[b][a] += Q / d

    return best_tour, best_distance

# cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(20)]
cities = Circle(10)[:, :2]
# cities = datas[0][:, 1:]
best_path, best_dist = ant_colony_tsp(cities)
print("Best tour:", best_path)
print("Best distance:", best_dist)

# import matplotlib.pyplot as plt

# x = [city[0] for city in cities]
# y = [city[1] for city in cities]
# plt.scatter(x, y)

# for i in range(len(best_path)):
#     j = (i + 1) % len(best_path)
#     plt.plot([cities[best_path[i]][0], cities[best_path[j]][0]], [cities[best_path[i]][1], cities[best_path[j]][1]], 'r-')

# plt.show()
