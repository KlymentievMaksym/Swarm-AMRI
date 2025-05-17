import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    from PlotSolo import Plot
else:
    from .PlotSolo import Plot


def DE(pop_size, iterations, function, limits, F: float = None, P: float = None, parameters_to_pass=None, **kwargs):
    plot_do = kwargs.get("plot", False)
    return_more = kwargs.get("more", False)
    every = kwargs.get("every", 1)

    do_random_F = F is None
    do_random_P = P is None

    dim = len(limits)
    limits = np.array(limits)
    # print(limits)
    x_low = limits[:, 0]
    x_high = limits[:, 1]

    population = np.random.uniform(x_low, x_high, (pop_size, dim))

    if plot_do or return_more:
        history_f = np.zeros((iterations//every, pop_size))
        history_pops = np.zeros((iterations//every, pop_size, dim))

        history_best_f = np.zeros(iterations//every)
        history_best_pop = np.zeros((iterations//every, dim))

    if parameters_to_pass is None:
        fitness = np.apply_along_axis(function, 1, population)
    else:
        fitness = np.apply_along_axis(function, 1, population, *parameters_to_pass)

    best_idx = np.argmin(fitness)
    best_f = fitness[best_idx]
    best_pop = population[best_idx].copy()
    max_f = np.max(fitness)

    for iteration in tqdm(
        range(iterations),
        desc="Processing",
        unit="step",
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour='cyan',
        total=iterations
    ):
        for i in range(pop_size):
            if do_random_F:
                F = np.random.uniform(1e-6, 2)
            if do_random_P:
                P = np.random.uniform(1e-6, 1)
            r = np.random.uniform(1e-6, 1, dim)

            idxs = np.delete(np.arange(pop_size), i)
            x1, x2, x3 = population[np.random.choice(idxs, 3, replace=False)]

            mutant_vector = x1 + F * (x2 - x3)
            mutant_vector = np.where(r < P, population[i], mutant_vector)
            mutant_vector = np.clip(mutant_vector, x_low, x_high)

            if parameters_to_pass is None:
                mutant_fitness = function(mutant_vector)
            else:
                mutant_fitness = function(mutant_vector, *parameters_to_pass)

            if fitness[i] > mutant_fitness:
                fitness[i] = mutant_fitness
                population[i] = mutant_vector.copy()

        el_min = np.argmin(fitness)
        current_best_f = fitness[el_min]
        if best_f > current_best_f:
            best_f = current_best_f
            best_pop = population[el_min].copy()

        max_f = max(max_f, np.max(fitness))

        if plot_do or return_more:
            if iteration % every == 0:
                history_best_f[iteration//every] = best_f
                history_best_pop[iteration//every] = best_pop.copy()
                history_f[iteration//every] = fitness.copy()
                history_pops[iteration//every] = population.copy()

    if plot_do:
        Plot(history_f, history_pops, history_best_f, history_best_pop, max_f, best_f, function, limits, parameters_to_pass, **kwargs)

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

    de = DE(100, 100, func, func_limit, plot=True, d3=True, d2=True, static=False, every=5)
    print(de)
