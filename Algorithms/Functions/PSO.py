import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    from PlotSolo import Plot
else:
    from .PlotSolo import Plot


def PSO(pop_size, iterations, random_limits, limits_speed, function, limits, a1: float = None, a2: float = None, parameters_to_pass=None, **kwargs):
    plot_do = kwargs.get("plot", False)
    return_more = kwargs.get("more", False)
    every = kwargs.get("every", 1)

    limits = np.array(limits)

    do_random_a1 = a1 is None
    do_random_a2 = a2 is None
    [low, high] = random_limits

    dim = len(limits)

    ints = kwargs.get("ints", [])

    kwargs = kwargs

    x_low = limits[:, 0]
    x_high = limits[:, 1]

    parts = np.random.uniform(x_low, x_high, (pop_size, dim))
    for i in ints:
        parts[:, i] = np.round(parts[:, i])

    if parameters_to_pass is None:
        fitness_func = np.apply_along_axis(function, 1, parts)
    else:
        fitness_func = np.apply_along_axis(function, 1, parts, *parameters_to_pass)

    max_f = 0

    index = np.argmin(fitness_func)
    best = fitness_func[index]
    best_dep_val = parts[index]

    if plot_do or return_more:
        history_parts = np.zeros((iterations//every, pop_size, dim))
        history_fitness_func = np.zeros((iterations//every, pop_size))

        history_best_dep_val = np.zeros((iterations//every, dim))
        history_best = np.zeros(iterations//every)

    speed = np.random.uniform(limits_speed[0], limits_speed[1], (pop_size, dim))

    best_personal = [float("inf") for part in range(pop_size)]
    best_personal_dep_val = [[0 for i in range(dim)] for j in range(pop_size)]

    first_time = True
    # run_before(**kwargs)
    for iteration in tqdm(
        range(iterations),
        desc="Processing",
        unit="step",
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour='cyan',
        total=iterations
    ):
        for i in range(pop_size):
            if do_random_a1:
                a1 = np.random.uniform(low, high)
            if do_random_a2:
                a2 = np.random.uniform(low, high)
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            if not first_time:
                if parameters_to_pass is None:
                    fitness_func[i] = function(parts[i])
                else:
                    fitness_func[i] = function(parts[i], *parameters_to_pass)

            if best_personal[i] > fitness_func[i]:
                best_personal[i] = fitness_func[i]
                best_personal_dep_val[i] = parts[i].copy()

            if best > fitness_func[i]:
                best = fitness_func[i]
                best_dep_val = parts[i].copy()

            to_best_self = (best_personal_dep_val[i] - parts[i]) * r1
            to_best_overal = (best_dep_val - parts[i]) * r2
            speed[i] = speed[i] + a1 * to_best_self + a2 * to_best_overal

            speed[i][speed[i] > limits_speed[1]] = limits_speed[1]
            speed[i][speed[i] < limits_speed[0]] = limits_speed[0]

            parts[i] = parts[i] + speed[i]
            for l in range(dim):
                if parts[i][l] > x_high[l]:
                    parts[i][l] = x_high[l] - abs(parts[i][l] - x_high[l])
                    speed[i][l] = -speed[i][l]
                elif parts[i][l] < x_low[l]:
                    parts[i][l] = x_low[l] + abs(parts[i][l] - x_low[l])
                    speed[i][l] = -speed[i][l]
            for it in ints:
                parts[i, it] = np.round(parts[i, it])

        el_max = np.max(fitness_func)
        if max_f < el_max:
            max_f = el_max

        if plot_do or return_more:
            if iteration % every == 0:
                history_fitness_func[iteration//every] = fitness_func.copy()
                history_parts[iteration//every] = parts.copy()

                history_best[iteration//every] = best
                history_best_dep_val[iteration//every] = best_dep_val.copy()
        first_time = False

    if plot_do or return_more:
        history_fitness_func[-1] = fitness_func.copy()
        history_parts[-1] = parts.copy()

        history_best[-1] = best
        history_best_dep_val[-1] = best_dep_val.copy()

    if plot_do:
        Plot(history_fitness_func, history_parts, history_best, history_best_dep_val, max_f, best, function, limits, parameters_to_pass, **kwargs)

    if return_more:
        return best, best_dep_val, history_fitness_func, history_parts, history_best, history_best_dep_val
    return best, best_dep_val
    # return run_after


if __name__ == "__main__":
    def F(X):
        A = 10
        length = len(X)
        result = A*length
        for x in X:
            result += x**2-A*np.cos(2*np.pi*x)
        return result
    F_limits = [[-5.12, 5.12], [-5.12, 5.12]]
    pso = PSO(40, 700, [0, 4], [-.15, .15], F, F_limits, plot=True, d2=True, d3=False, static=False, every=5)
