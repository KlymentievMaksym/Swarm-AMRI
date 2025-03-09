import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class FF:
    def __init__(self, pop_size, iterations, limits_speed, function, limits, **kwargs):
        self.iterations = iterations
        self.pop_size = pop_size
        self.function = function
        self.limits = limits

        self.dim = len(self.limits)

        self.integer = kwargs.get("integer", [])

        self.x_low = [limit[0] for limit in self.limits]
        self.x_high = [limit[1] for limit in self.limits]
        self.parts = np.random.uniform(self.x_low, self.x_high, (self.pop_size, self.dim))
        for i in self.integer:
            self.parts[:, i] = np.round(self.parts[:, i])

        self.best_personal = [float("inf") for part in range(self.pop_size)]
        self.best_personal_dep_val = [[0 for i in range(self.dim)] for j in range(self.pop_size)]

        self.best = float("inf")
        self.best_dep_val = [0 for i in range(self.dim)]

        self.history_parts = []
        self.history_fitness_func = []

        self.history_best_dep_val = []
        self.history_best = []

    def run(self, **kwargs):
        self.plot(**kwargs)
        return self.best, self.best_dep_val

    def plot(self, **kwargs):
        dots = kwargs.get("dots", 500)

        d2 = kwargs.get("d2", False)
        d3 = kwargs.get("d3", False)

        save_path = kwargs.get("save", None)

        show = kwargs.get("show", False)

        if self.dim <= 2 and (show or save_path is not None):
            self.projection_dep_val = np.linspace(self.x_low, self.x_high, dots)
            space = np.array([self.projection_dep_val[:, i] for i in range(self.dim)])
            space = np.meshgrid(*space)
            self.projection = np.array([[self.function([space[i][j, k] for i in range(self.dim)]) for k in range(dots)] for j in range(dots)])

            if d2:
                fig, ax1 = plt.subplots()
            elif d3:
                fig = plt.figure(figsize=plt.figaspect(2.))
                ax1 = fig.add_subplot(1, 1, 1, projection='3d')
            if (d2 or d3):
                cs = ax1.contourf(*space, self.projection, cmap="cool")
                fig.colorbar(cs)

                # print(self.history_best)
                def update(frame):
                    ax1.clear()
                    ax1.set_title(f"Best solution: {self.history_best[frame]:.5f} | Best dep val: {self.history_best_dep_val[frame]} | Iter {frame}")
                    if d2:
                        ax1.contourf(*space, self.projection, cmap="cool")
                        print(self.history_best_dep_val[frame], self.history_best[frame])
                        ax1.scatter(*[self.history_parts[frame][:, i] for i in range(self.dim)], label="Population", c='black')
                        ax1.scatter(*[self.history_best_dep_val[frame][i] for i in range(self.dim)], label="Best", c='yellow')
                    elif d3:
                        ax1.plot_surface(*space, self.projection, cmap="cool", alpha=0.8)
                        ax1.scatter(*[self.history_parts[frame][:, i] for i in range(self.dim)], self.history_fitness_func[frame], label="Population", c='Black')
                        ax1.scatter(*[self.history_best_dep_val[frame][i] for i in range(self.dim)], self.history_best[frame], label="Best", c='Red')
                    ax1.legend()

                ani = animation.FuncAnimation(fig=fig, func=update, frames=self.iterations, interval=30)
                if save_path is not None:
                    # print("Saving")
                    ani.save(save_path, fps=10)
                if show:
                    fig.canvas.manager.window.state('zoomed')
                    plt.show()