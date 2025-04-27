import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    from Generator import Generator
    from Plot import Plot
else:
    from .Plot import Plot


def fitness(population, weight, values, max_capacity):
    total_weight = np.sum(population * weight)
    if total_weight > max_capacity:
        return 0
    total_value = np.sum(population * values)
    return -total_value


def crossover(parent1, parent2):
    cross = np.random.randint(1, parent1.shape[0])
    child = np.append(parent1[:cross], parent2[cross:], axis=0)
    return child


def mutation(child):
    i = np.random.randint(child.shape[0])
    j = np.random.randint(child.shape[0])
    while i == j:
        j = np.random.randint(child.shape[0])
    if i > j:
        i, j = j, i
    if np.random.rand() < 0.5:
        child[i], child[j] = child[j], child[i]
    else:
        # print(child[i:j])
        # print(child[i:j][::-1])
        child[i:j] = child[i:j][::-1]
    return child


def Genetic(iterations: int, pop_size: int, child_size: int, mutation_probability: float, max_capacity: float, weight_price: list[list[float, float]], **kwargs):
    every = kwargs.get("every", 1)

    fitnes = np.zeros(pop_size + child_size)
    best_f = 0
    best_items_choose = np.zeros((weight_price.shape[0]))

    if isinstance(weight_price, list):
        weight_price = np.array(weight_price)
    if not isinstance(weight_price, np.ndarray):
        raise Exception("weight_price must be a list or numpy array")
    population = np.zeros((pop_size + child_size, weight_price.shape[0]))
    population[:pop_size] = np.random.randint(0, 2, size=(pop_size, weight_price.shape[0]))

    history_pop = np.zeros((iterations//every + 1, pop_size + child_size, weight_price.shape[0]))
    history_fitness = np.zeros((iterations//every + 1, pop_size + child_size))

    for iteration in tqdm(
        range(iterations),
        desc="Processing",
        unit="step",
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour='cyan'
    ):
        for child_index in range(pop_size, pop_size + child_size):
            i = np.random.randint(pop_size)
            j = np.random.randint(pop_size)
            while i == j:
                j = np.random.randint(pop_size)
            if i > j:
                i, j = j, i
            population[child_index] = crossover(population[i], population[j])
            if np.random.rand() < mutation_probability:
                population[child_index] = mutation(population[child_index])

        for pupil in range(pop_size + child_size):
            fitnes[pupil] = fitness(population[pupil], weight_price[:, 0], weight_price[:, 1], max_capacity)

        ind = np.argsort(fitnes)
        fitnes = fitnes[ind]
        if fitnes[0] == 0 and fitnes[-1] == 0:
            print("Same fitnes. Restarting population")
            population[:pop_size] = np.random.randint(0, 2, size=(pop_size, weight_price.shape[0]))
        else:
            population = population[ind]

        if best_f < abs(fitnes[0]):
            best_f = abs(fitnes[0])
            best_items_choose = population[0]
        if (iteration + 1) % every == 0:
            history_pop[iteration//every] = population.copy()
            history_fitness[iteration//every] = -fitnes.copy()
    history_pop[-1] = population.copy()
    history_fitness[-1] = -fitnes.copy()
    Plot().Plot(history_pop, history_fitness, **kwargs)
    return best_f, best_items_choose


if __name__ == "__main__":
    weigth_max, data = Generator(100, min_amount=20, max_price=10, max_weight=10).generate()
    best_f, best_items_choose = Genetic(100, 20, 40, 0.5, weigth_max, data, show_convergence_animation=True, show_bar_animation=True, every=10, interval=3000)  # , save_convergence="convergence.gif"
    print(best_f, best_items_choose)
    print(f"Max capacity: {weigth_max}, Weight used: {np.sum(data[:, 0] * best_items_choose)}, Best value: {best_f}\nBest items choose: \n{best_items_choose}")
    # print(f"Max theoretical capacity: {np.sum(data[:, 0])}, Max theoretical value: {np.sum(data[:, 1])}")
    print(f"Weight: \n{data[:, 0]}\nValue: \n{data[:, 1]}")
