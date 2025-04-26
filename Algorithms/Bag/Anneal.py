import numpy as np


if __name__ == "__main__":
    from Generator import Generator
# else:
#     from .Generator import Generator


def fitness(population, weight, values, max_capacity):
    total_weight = np.sum(population * weight)
    if total_weight > max_capacity:
        return 0
    total_value = np.sum(population * values)
    return total_value


def mutate(route):
    solution_route = route.copy()
    i = np.random.randint(solution_route.shape[0])
    j = np.random.randint(solution_route.shape[0])
    while i == j:
        j = np.random.randint(solution_route.shape[0])
    if i > j:
        i, j = j, i
    choosen_mutatuion = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
    match choosen_mutatuion:
        case 0:
            if np.random.rand() < 0.5:
                solution_route[i] = 1 - solution_route[i]
            else:
                solution_route[j] = 1 - solution_route[j]
        case 1:
            solution_route[i], solution_route[j] = solution_route[j], solution_route[i]
        case 2:
            solution_route[i:j] = solution_route[i:j][::-1]
    return solution_route


def Anneal(
    iterations: int,
    T_min: float,
    T_max: float,
    cooling: float,
    weight_price: np.ndarray,
    max_capacity: float,
):
    T = T_max
    weight = weight_price[:, 0]
    price = weight_price[:, 1]

    solution_route = np.random.randint(0, 2, size=(weight_price.shape[0]))
    solution = fitness(solution_route, weight, price, max_capacity)

    for iteration in range(iterations):
        solution_main = fitness(solution_route, weight, price, max_capacity)
        neighbor_route = mutate(solution_route)
        solution_neighbor = fitness(neighbor_route, weight, price, max_capacity)
        delta = solution_main - solution_neighbor
        if delta < 0:
            # if best_solution < solution_neighbor:
            #     best_solution = solution_neighbor
            #     best_items = neighbor_route
            solution_route = neighbor_route
            solution = solution_neighbor
        else:
            if np.random.rand() < np.exp(-delta / T):
                # if best_solution < solution_neighbor:
                #     best_solution = solution_neighbor
                #     best_items = neighbor_route
                solution_route = neighbor_route
                solution = solution_neighbor
        T *= cooling
        if T < T_min:
            break
    # return best_solution, best_items
    return solution, solution_route


if __name__ == "__main__":
    capacity, data = Generator(10, min_amount=10, max_price=10, max_weight=10).generate()
    solution, solution_items = Anneal(1000, 1e-7, 100, 0.99, data, capacity)
    print("Best solution:", solution)
    print("Items:", solution_items)
    print("Weigh:", data[:, 0].astype(int))
    print("Price:", data[:, 1].astype(int))
    print("Total weight:", np.sum(solution_items * data[:, 0]))
    print("Max weight:", capacity)
