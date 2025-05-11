import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


from typing import Callable

from time import time


class Algorithm:
    def __init__(self, pop_size: int, iterations: int, random_limits: list[list[float, float]], function: Callable, limits: list[list[float, float]], **kwargs):
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
        # self.fitness_func = np.array([self.function(part) for part in self.parts])
        self.fitness_func = np.apply_along_axis(self.function, 1, self.parts)

        self.best = np.min(self.fitness_func)
        self.best_dep_val = self.parts[np.argmin(self.fitness_func)]
        # self.best = float("inf")
        # self.best_dep_val = [0 for i in range(self.dim)]

        self.history_parts = []
        self.history_fitness_func = []

        self.history_best_dep_val = []
        self.history_best = []

        self.same = False

        self.time_smooth = max(10, int(0.01*self.iterations))
        self.time_hist = np.zeros(self.time_smooth)
        self.time_index = 0

    def run_before(self, **kwargs):
        self.kwargs.update(kwargs)
        self.define_kwargs
        self.prev_time = time()

    @property
    def define_kwargs(self):
        self.d1 = self.kwargs.get("d1", False)
        self.d2 = self.kwargs.get("d2", False)
        self.d3 = self.kwargs.get("d3", False)

        self.save_path = self.kwargs.get("save", None)
        self.save_path_photo = self.kwargs.get("savep", None)
        self.fps = self.kwargs.get("fps", 30)
        self.interval = self.kwargs.get("interval", 30)

        self.progress = self.kwargs.get("progress", True)
        self.show = self.kwargs.get("show", False)
        self.plot_do = self.kwargs.get("plot", True)

        self.possible_styles = {"linspace": np.linspace, "arange": np.arange}
        self.style = self.kwargs.get("style", "linspace")
        if self.style not in self.possible_styles:
            raise Exception(f"{self.style} does not exists")
        self.dots = self.kwargs.get("dots", 500) if self.style == "linspace" else self.kwargs.get("dots", 0.1)

        self.close = self.kwargs.get("close", True)
        self.label = self.kwargs.get("label", self.__class__.__name__)

        self.history = self.kwargs.get("history", False)

        self.cnt = 0
        self.cnt_max = self.kwargs.get("count", 3)
        self.break_faster = self.kwargs.get("break_faster", False)
        self.epsilon = self.kwargs.get("epsilon", 1e-6)

        self.show_all_population = self.kwargs.get("population", True)

        self.random = self.kwargs.get("random", True)

        self.possible_formats = {"hms": ["h ", "m ", "s "], ":": [":", ":", ""]}
        self.format = self.kwargs.get("format", "hms")
        if self.format not in self.possible_formats:
            raise Exception(f"{self.format} does not exists")

    @property
    def run_after(self):
        if self.show or self.save_path is not None:
            self.plot(**self.kwargs)
        if self.history:
            return self.history_best, self.history_best_dep_val, self.history_fitness_func, self.history_parts
        return self.best, self.best_dep_val

    @property
    def check(self):
        if not self.same and self.best != float("inf") and len(self.history_best) != 0:
            self.check_if_same(self.best, self.history_best[-1])

    def check_if_same(self, prev_best, new_best):
        if abs(new_best - prev_best) < self.epsilon:
            self.cnt += 1
            if self.cnt == self.cnt_max:
                self.same = True
                return True
            return False
        self.cnt = 0
        return False

    @property
    def save(self):
        if self.show or self.save_path is not None or self.history:
            if self.show_all_population:
                self.history_parts.append(self.parts.copy())
                self.history_fitness_func.append(self.fitness_func.copy())
            else:
                self.history_parts.append(self.parts[:self.pop_size].copy())
                self.history_fitness_func.append(self.fitness_func[:self.pop_size].copy())

            self.history_best_dep_val.append(self.best_dep_val.copy())
            self.history_best.append(self.best)

    def progress_bar(self, quantity, total, **kwargs):
        self.new_time = time()
        if not self.progress:
            return
        name = kwargs.get("name", "No Name")
        percent = (quantity / total) * 100
        space = "                 "

        time_left = (self.new_time-self.prev_time)*(total - quantity)

        self.time_hist[self.time_index] = time_left
        self.time_index = (self.time_index + 1) % self.time_smooth
        time_left = np.mean(self.time_hist)

        time_left_txt = f"{(time_left//3600):.0f}{"{0}"}{(time_left//60 % 60):.0f}{"{1}"}{(time_left % 60):.0f}{"{2}"}"

        time_left_txt = time_left_txt.format(*self.possible_formats[self.format])
        quantity_txt = str(quantity).zfill(len(str(total)))

        print(f"Progress {name} : {quantity_txt}/{total} ({percent:.2f}%) | {time_left_txt}{space}", end='\r')

        self.prev_time = self.new_time
        if quantity == total-1 or (self.same and self.break_faster):
            percent = ((quantity+1) / total) * 100
            quantity_txt = str(quantity+1).zfill(len(str(total)))
            print(f"Progress {name} : {quantity_txt}/{total} ({percent:.2f}%){space*2}")
            # print()

    def plot(self, **kwargs):
        if self.show:
            # if type(self.history_best[0]) is not type(self.history_best[-1]):
            #     # print(type(self.history_best[0]), type(self.history_best[-1]))
            #     self.history_best[0] = np.array([self.history_best[0]])
            #     # print(type(self.history_best[0]), type(self.history_best[-1]))
            plt.plot(list(range(self.iterations))[:len(self.history_best)], self.history_best, label=self.label)
            plt.title(self.function.__name__)
            plt.xlabel("Iterations")
            plt.ylabel("Fitness func value")
            plt.grid(True)
            plt.legend()
            if self.save_path_photo is not None:
                plt.savefig(self.save_path_photo)
            if self.plot_do:
                plt.show()
            else:
                if self.close:
                    plt.close()

        if self.dim <= 2 and (self.show or self.save_path is not None):

            if self.d1 or self.d2:
                fig, ax1 = plt.subplots()
            elif self.d3 and self.dim == 2:
                fig = plt.figure(figsize=plt.figaspect(2.))
                ax1 = fig.add_subplot(1, 1, 1, projection='3d')
            if self.d1 and self.dim == 1:
                if self.style == "linspace":
                    self.projection_dep_val = np.linspace(self.x_low, self.x_high, self.dots)
                elif self.style == "arange":
                    self.projection_dep_val = np.arange(self.x_low, self.x_high, self.dots)
                self.projection = np.array([self.function(dot) for dot in self.projection_dep_val])
                ax1.plot(self.projection_dep_val, self.projection)
            elif (self.d2 or self.d3) and self.dim == 2:
                if self.style == "linspace":
                    self.projection_dep_val = np.linspace(self.x_low, self.x_high, self.dots)
                    space = np.array([self.projection_dep_val[:, i] for i in range(self.dim)])
                elif self.style == "arange":
                    self.projection_dep_val = np.array([np.arange(self.x_low[i], self.x_high[i], self.dots) for i in range(self.dim)])
                    space = self.projection_dep_val.copy()
                space = np.meshgrid(*space)
                if self.style == "linspace":
                    self.projection = np.array([[self.function([space[i][j, k] for i in range(self.dim)]) for k in range(self.dots)] for j in range(self.dots)])
                elif self.style == "arange":
                    self.projection = self.function(space)

                cs = ax1.contourf(*space, self.projection, cmap="cool")
                fig.colorbar(cs)

            if (self.d1 or self.d2 or self.d3):
                def update(frame):
                    ax1.clear()
                    try:
                        ax1.set_title(f"{self.function.__name__}\nIter {frame}\nBest solution: {self.history_best[frame]:.5f}\nBest dep val: {self.history_best_dep_val[frame]}")
                    except TypeError:
                        ax1.set_title(f"{self.function.__name__}\nIter {frame}\nBest solution: {self.history_best[frame]}\nBest dep val: {self.history_best_dep_val[frame]}")
                    if self.d1 and self.dim == 1:
                        ax1.plot(self.projection_dep_val, self.projection)
                        ax1.scatter(self.history_parts[frame], self.history_fitness_func[frame], label="Population", c='Black')
                        ax1.scatter(self.history_best_dep_val[frame], self.history_best[frame], label="Best", c='Red')
                    elif self.d2 and self.dim == 2:
                        ax1.contourf(*space, self.projection, cmap="cool")
                        # print(self.history_best_dep_val[frame], self.history_best[frame])
                        ax1.scatter(*[self.history_parts[frame][:, i] for i in range(self.dim)], label="Population", c='black')
                        ax1.scatter(*[self.history_best_dep_val[frame][i] for i in range(self.dim)], label="Best", c='yellow')
                    elif self.d3 and self.dim == 2:
                        ax1.plot_surface(*space, self.projection, cmap="cool", alpha=0.8)
                        ax1.scatter(*[self.history_parts[frame][:, i] for i in range(self.dim)], self.history_fitness_func[frame], label="Population", c='Black')
                        ax1.scatter(*[self.history_best_dep_val[frame][i] for i in range(self.dim)], self.history_best[frame], label="Best", c='Red')
                    ax1.legend()

                ani = animation.FuncAnimation(fig=fig, func=update, frames=min(self.iterations, len(self.history_best)), interval=self.interval)
                if self.save_path is not None:
                    ani.save(self.save_path, fps=self.fps)
                if self.plot_do:
                    fig.canvas.manager.window.state('zoomed')
                    plt.show()
