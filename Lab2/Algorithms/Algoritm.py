import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Algorithm:
    def __init__(self, pop_size, iterations, random_limits, function, limits, **kwargs):
        self.iterations = iterations
        self.pop_size = pop_size
        self.function = function
        self.limits = np.array(limits)

        [self.low, self.high] = random_limits

        self.dim = len(self.limits)

        self.integer = kwargs.get("integer", [])

        self.kwargs = kwargs

        # self.x_low = [limit[0] for limit in self.limits]
        # self.x_high = [limit[1] for limit in self.limits]
        self.x_low = self.limits[:, 0]
        self.x_high = self.limits[:, 1]
        self.parts = np.random.uniform(self.x_low, self.x_high, (self.pop_size, self.dim))
        for i in self.integer:
            self.parts[:, i] = np.round(self.parts[:, i])
        self.fitness_func = np.array([self.function(part) for part in self.parts])

        self.best = float("inf")
        self.best_dep_val = [0 for i in range(self.dim)]

        self.history_parts = []
        self.history_fitness_func = []

        self.history_best_dep_val = []
        self.history_best = []

        self.same = False

        self.cnt_max = kwargs.get("count", 3)
        self.cnt = 0
        self.epsilon = kwargs.get("epsilon", 0.0000001)
        self.progress = kwargs.get("progress", True)

    def check_if_same(self, prev_best, new_best):
        if abs(new_best - prev_best) < self.epsilon:
            self.cnt += 1
            if self.cnt == self.cnt_max:
                self.same = True
                return True
            return False
        self.cnt = 0
        return False

    def progress_bar(self, quantity, total, **kwargs):
        if not self.progress:
            return
        name = kwargs.get("name", "No Name")
        percent = (quantity / total) * 100
        print(f"Progress {name} : {quantity}/{total} ({percent:.2f}%)", end='\r')
        if quantity == total-1:
            percent = ((quantity+1) / total) * 100
            print(f"Progress {name} : {quantity+1}/{total} ({percent:.2f}%)")
            # print()

    def plot(self, **kwargs):
        dots = kwargs.get("dots", 500)

        d2 = kwargs.get("d2", False)
        d3 = kwargs.get("d3", False)

        save_path = kwargs.get("save", None)

        show = kwargs.get("show", False)
        plot = kwargs.get("plot", True)

        if show:
            plt.plot(list(range(self.iterations))[:len(self.history_best)], self.history_best, label=self.__class__.__name__)
            plt.grid()
            plt.legend()
            if plot:
                plt.show()

        if self.dim <= 2 and (show or save_path is not None):
            if d2:
                fig, ax1 = plt.subplots()
            elif d3:
                fig = plt.figure(figsize=plt.figaspect(2.))
                ax1 = fig.add_subplot(1, 1, 1, projection='3d')
            if (d2 or d3):
                self.projection_dep_val = np.linspace(self.x_low, self.x_high, dots)
                space = np.array([self.projection_dep_val[:, i] for i in range(self.dim)])
                space = np.meshgrid(*space)
                self.projection = np.array([[self.function([space[i][j, k] for i in range(self.dim)]) for k in range(dots)] for j in range(dots)])

                cs = ax1.contourf(*space, self.projection, cmap="cool")
                fig.colorbar(cs)

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

                ani = animation.FuncAnimation(fig=fig, func=update, frames=min(self.iterations, len(self.history_best)), interval=10)
                if save_path is not None:
                    ani.save(save_path, fps=10)
                if show:
                    fig.canvas.manager.window.state('zoomed')
                    plt.show()