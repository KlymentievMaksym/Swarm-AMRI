
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec


def _set_limits(ax1, ax2, min_f, max_f, history_f):
    ax1.set_xlim([-1, len(history_f[0])])
    ax2.set_xlim([-1, len(history_f)])
    if abs(max_f) != float('inf') and abs(min_f) != float('inf'):
        ax1.set_ylim([min_f - 1, max_f + 1])
        ax2.set_ylim([min_f - 1, max_f + 1])
    else:
        check = np.abs(history_f) != float('inf')
        # print(check)
        not_inf_data = history_f[check]
        # print(not_inf_data)
        ax1.set_ylim([np.min(not_inf_data) - 1, np.max(not_inf_data) + 1])
        ax2.set_ylim([np.min(not_inf_data) - 1, np.max(not_inf_data) + 1])

    ax1.set_xlabel("Pop Part")
    ax1.set_ylabel("Fitness")

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Fitness")

    ax1.legend()
    ax2.legend()

    ax1.grid(True)
    ax2.grid(True)

    return ax1, ax2


def _update(frame, pop_fitness, best_fitness, history_best_f, history_f):
    plt.suptitle(f"Best solution: {history_best_f[frame]:.5f} | Iter {frame}")
    pop_fitness.set_xdata(list(range(len(history_f[frame]))))
    pop_fitness.set_ydata(history_f[frame])

    best_fitness.set_xdata(list(range(len(history_best_f[:frame]))))
    best_fitness.set_ydata(history_best_f[:frame])

    return pop_fitness, best_fitness


def _create_axes(frame, ax1, ax2, history_f, history_best_f, min_f, max_f, inf, static):
    pop_fitness = ax1.plot(history_f[frame], label="Fitness")[0]
    best_fitness = ax2.plot(history_best_f[:frame], label="Best")[0]

    if static:
        plt.suptitle(f"Best solution: {history_best_f[frame]:.5f}")
        if inf:
            max_f = np.max(history_best_f)
            min_f = np.min(history_best_f)
        # print(max_f, min_f)
        _set_limits(ax1, ax2, min_f, max_f, history_f)

    return pop_fitness, best_fitness


def _create_axes_d2_d3(frame, fig, ax3, ax4, space, projection, history_f, history_pops, history_best_f, history_best_pop, pop, best):
    contourf = ax3.contourf(*space, projection)
    pop_scatter = ax3.scatter(history_pops[frame, :, 0], history_pops[frame, :, 1], label="Population", color=pop, alpha=.7)
    best_scatter = ax3.scatter(history_best_pop[frame, 0], history_best_pop[frame, 1], label="Best", color=best, s=50)
    fig.colorbar(contourf, ax=ax3)

    contourf_3d = ax4.plot_wireframe(*space, projection, color="gray", alpha=.1)
    pop_scatter_3d = ax4.scatter(history_pops[frame, :, 0], history_pops[frame, :, 1], history_f[frame], label="Population", color=pop, alpha=.7)
    best_scatter_3d = ax4.scatter(history_best_pop[frame, 0], history_best_pop[frame, 1], history_best_f[frame], label="Best", color=best, s=50)

    return pop_scatter, best_scatter, pop_scatter_3d, best_scatter_3d


def _create_axes_d2(frame, fig, ax3, space, projection, history_pops, history_best_pop, pop, best):
    contourf = ax3.contourf(*space, projection)
    pop_scatter = ax3.scatter(history_pops[frame, :, 0], history_pops[frame, :, 1], label="Population", color=pop, alpha=.7)
    best_scatter = ax3.scatter(history_best_pop[frame, 0], history_best_pop[frame, 1], label="Best", color=best, s=50)
    fig.colorbar(contourf, ax=ax3)

    return pop_scatter, best_scatter


def _create_axes_d3(frame, ax3, space, projection, history_f, history_pops, history_best_f, history_best_pop, pop, best):
    contourf_3d = ax3.plot_wireframe(*space, projection, color="gray", alpha=.1)
    pop_scatter_3d = ax3.scatter(history_pops[frame, :, 0], history_pops[frame, :, 1], history_f[frame], label="Population", color=pop, alpha=.7)
    best_scatter_3d = ax3.scatter(history_best_pop[frame, 0], history_best_pop[frame, 1], history_best_f[frame], label="Best", color=best, s=50)

    return pop_scatter_3d, best_scatter_3d


def Plot(history_f, history_pops, history_best_f, history_best_pop, max_f, min_f, function, limits, **kwargs):
    d2 = kwargs.get("d2", True)
    d3 = kwargs.get("d3", False)
    save = kwargs.get("save", False)
    static = kwargs.get("static", False)
    inf = kwargs.get("inf", False)

    pop = "LightBlue"
    best = "Red"

    if limits.shape[0] != 2 or (not d2 and not d3):
        fig, [ax1, ax2] = plt.subplots(2, figsize=(8, 6))

        if static:
            frame = -1
            plt.suptitle(f"Best solution: {history_best_f[frame]:.5f} | Iter {len(history_best_f)}")
            pop_fitness, best_fitness = _create_axes(frame, ax1, ax2, history_f, history_best_f, min_f, max_f, inf, static)
        else:
            frame = 0
            pop_fitness, best_fitness = _create_axes(frame, ax1, ax2, history_f, history_best_f, min_f, max_f, inf, static)

            def init():
                _set_limits(ax1, ax2, min_f, max_f, history_f)

            def update(frame):
                _update(frame, pop_fitness, best_fitness, history_best_f, history_f)
                return (pop_fitness, best_fitness)
    else:
        fig = plt.figure(figsize=(8, 6))
        if d3:
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        else:
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 2])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        if d2 and d3:
            ax3 = fig.add_subplot(gs[0, 1])
            ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        elif d2:
            ax3 = fig.add_subplot(gs[:, 1])
        elif d3:
            ax3 = fig.add_subplot(gs[:, 1], projection='3d')

        x_low, x_high = limits[:, 0], limits[:, 1]
        dim = len(limits)
        dots = 200
        projection_dep_val = np.linspace(x_low, x_high, dots)
        space = np.array([projection_dep_val[:, i] for i in range(dim)])
        space = np.meshgrid(*space)
        projection = np.array([[function(np.array([space[i][j, k] for i in range(dim)])) for k in range(dots)] for j in range(dots)])

        if static:
            frame = -1
            pop_fitness, best_fitness = _create_axes(frame, ax1, ax2, history_f, history_best_f, min_f, max_f, inf, static)

            if d2 and d3:
                pop_scatter, best_scatter, pop_scatter_3d, best_scatter_3d = _create_axes_d2_d3(frame, fig, ax3, ax4, space, projection, history_f, history_pops, history_best_f, history_best_pop, pop, best)
            elif d2:
                pop_scatter, best_scatter = _create_axes_d2(frame, fig, ax3, space, projection, history_pops, history_best_pop, pop, best)
            elif d3:
                pop_scatter_3d, best_scatter_3d = _create_axes_d3(frame, ax3, space, projection, history_f, history_pops, history_best_f, history_best_pop, pop, best)

        else:
            frame = 0
            pop_fitness, best_fitness = _create_axes(frame, ax1, ax2, history_f, history_best_f, min_f, max_f, inf, static)

            if d2 and d3:
                pop_scatter, best_scatter, pop_scatter_3d, best_scatter_3d = _create_axes_d2_d3(frame, fig, ax3, ax4, space, projection, history_f, history_pops, history_best_f, history_best_pop, pop, best)
            elif d2:
                pop_scatter, best_scatter = _create_axes_d2(frame, fig, ax3, space, projection, history_pops, history_best_pop, pop, best)
            elif d3:
                pop_scatter_3d, best_scatter_3d = _create_axes_d3(frame, ax3, space, projection, history_f, history_pops, history_best_f, history_best_pop, pop, best)

            def init():
                _set_limits(ax1, ax2, min_f, max_f, history_f)
                ax3.legend()
                if d2 and d3:
                    ax4.legend()

                ax3.grid(True)
                if d2 and d3:
                    ax4.grid(True)

                    ax4.view_init(elev=45, azim=35)
                elif d3:
                    ax3.view_init(elev=45, azim=35)

            def update(frame):
                _update(frame, pop_fitness, best_fitness, history_best_f, history_f)

                if d2:
                    pop_scatter.set_offsets(history_pops[frame, :])
                    best_scatter.set_offsets(history_best_pop[frame])

                if d3:
                    pop_scatter_3d._offsets3d = (
                        history_pops[frame, :, 0],
                        history_pops[frame, :, 1],
                        history_f[frame]
                    )
                    best_scatter_3d._offsets3d = (
                        [history_best_pop[frame, 0]],
                        [history_best_pop[frame, 1]],
                        [history_best_f[frame]]
                    )

                if d2 and d3:
                    return (pop_fitness, best_fitness, pop_scatter, best_scatter, pop_scatter_3d, best_scatter_3d)
                elif d2:
                    return (pop_fitness, best_fitness, pop_scatter, best_scatter)
                elif d3:
                    return (pop_fitness, best_fitness, pop_scatter_3d, best_scatter_3d)
                return (pop_fitness, best_fitness)

    if not static:
        anim = animation.FuncAnimation(fig, update, frames=len(history_f), init_func=init, interval=100)
        if save:
            try:
                anim.save(save)
            except Exception as e:
                print(e)
    plt.show()
