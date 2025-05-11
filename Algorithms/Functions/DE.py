import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    from PlotSolo import Plot
else:
    from .PlotSolo import Plot


def DE(pop_size, iterations, function, limits, **kwargs):
    plot_do = kwargs.get("plot", False)
    return_more = kwargs.get("more", False)
    dim = len(limits)
    limits = np.array(limits)
    # print(limits)
    x_low = limits[:, 0]
    x_high = limits[:, 1]

    population = np.random.uniform(x_low, x_high, (pop_size, dim))

    max_f = 0
    best_f = float('inf')
    best_pop = np.zeros(dim)

    if plot_do:
        history_f = np.zeros((iterations, pop_size))
        history_pops = np.zeros((iterations, pop_size, dim))

        history_best_f = np.zeros(iterations)
        history_best_pop = np.zeros((iterations, dim))

    fitness = np.array([function(X) for X in population])
    for iteration in tqdm(
        range(iterations),
        desc="Processing",
        unit="step",
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour='cyan',
        total=iterations
    ):
        for i in range(pop_size):
            F = np.random.uniform(1e-6, 2)
            P = np.random.uniform(1e-6, 1)
            r = np.random.uniform(1e-6, 1, dim)

            x1, x2, x3 = np.random.choice(population.shape[0], size=3, replace=False)
            while np.all(population[x1] == population[i]) or np.all(population[x2] == population[i]) or np.all(population[x3] == population[i]):
                x1, x2, x3 = np.random.choice(population.shape[0], size=3, replace=False)

            if np.random.rand() < 0.5:
                mutant_vect = population[x1] + F * (population[x2] - population[x3])
                mutant_vector = population[i].copy()
                mutant_vector[r < P] = mutant_vect[r < P]
            else:
                mutant_vector = population[x1] + F * (population[x2] - population[x3])
                mutant_vector[r < P] = population[i][r < P]

            mutant_vector = np.clip(mutant_vector, x_low, x_high)
            mutant_fitness = function(mutant_vector)
            if fitness[i] > mutant_fitness:
                fitness[i] = mutant_fitness
                population[i] = mutant_vector.copy()

        el_min = np.argmin(fitness)
        if best_f > fitness[el_min]:
            best_f = fitness[el_min]
            best_pop = population[el_min].copy()

        el_max = np.max(fitness)
        if max_f < el_max:
            max_f = el_max

        if plot_do:
            history_best_f[iteration] = best_f
            history_best_pop[iteration] = best_pop.copy()
            history_f[iteration] = fitness.copy()
            history_pops[iteration] = population.copy()

    if plot_do:
        Plot(history_f, history_pops, history_best_f, history_best_pop, max_f, best_f, function, limits, **kwargs)

    if return_more:
        return best_f, best_pop, history_f, history_pops, history_best_f, history_best_pop
    return best_f, best_pop


if __name__ == "__main__":
    def func(X):
        return np.sum(X**2)
    func_limit = [[-1e2, 1e2]] * 2

    def func(X):
        A = 10
        length = len(X)
        result = A*length
        for x in X:
            result += x**2-A*np.cos(2*np.pi*x)
        return result
    func_limit = [[-5.12, 5.12], [-5.12, 5.12]]

    # def func(X):
    #     x1, x2, x3, x4, x5, x6, x7, = X
    #     f1 = 27 / (x1 * x2**2 * x3) - 1  <= 0
    #     f2 = 397.5 / (x1 * x2**2 * x3**2) - 1 <= 0
    #     f3 = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1 <= 0
    #     f4 = 1.93 / (x2 * x3 * x7**4) - 1 <= 0
    #     f5 = 1.0/(110 * x6**3) * np.sqrt(((745*x4) / (x2 * x3))**2 + 16.9 * 10**6) - 1 <= 0
    #     f6 = 1.0/(85 * x7**3) * np.sqrt(((745*x5) / (x2 * x3))**2 + 157.5 * 10**6) - 1 <= 0
    #     f7 = (x2*x3) / 40 - 1 <= 0
    #     f8 = 5*x2 / x1 - 1 <= 0
    #     f9 = x1 / (12 * x2) - 1 <= 0
    #     f10 = (1.5 * x6 + 1.9) / x4 - 1 <= 0
    #     f11 = (1.1 * x7 + 1.9) / x5 - 1 <= 0
    #     if f1 and f2 and f3 and f4 and f5 and f6 and f7 and f8 and f9 and f10 and f11:
    #         return 0.7854*x1*x2**2*(3.3333*x3**2 + 14.9334*x3 - 43.0934) - 1.508*x1*(x6**2 + x7**2) + 7.4777*(x6**3 + x7**3) + 0.7854*(x4*x6**2 + x5*x7**2)
    #     return float('inf')
    # func_limit = [[2.6, 3.6], [0.7, 0.8], [17, 28], [7.3, 8.3], [7.8, 8.3], [2.9, 3.9], [5.0, 5.5]]

    de = DE(100, 100, func, func_limit, plot=True, d3=True, d2=True, static=True)
    print(de)
