import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class BEE:
    def __init__(self, pop_size, iterations, areas_num, areas_elite_num, bees_depart_area_num, bees_depart_area_elite_num, function, limits, **kwargs):
        self.pop_size = pop_size
        self.iterations = iterations

        self.areas_num = areas_num
        self.areas_elite_num = areas_elite_num
        self.bees_depart_area_num = bees_depart_area_num
        self.bees_depart_area_elite_num = bees_depart_area_elite_num

        self.function = function
        self.limits = limits

        self.dim = len(self.limits)

        self.integer = kwargs.get("integer", [])
        self.kwargs = kwargs

        self.x_low = [limit[0] for limit in self.limits]
        self.x_high = [limit[1] for limit in self.limits]
        self.bees = np.random.uniform(self.x_low, self.x_high, (self.pop_size, self.dim))
        self.fitness_func = np.array([self.function(part) for part in self.bees])

        for i in self.integer:
            self.bees[:, i] = np.round(self.bees[:, i])

        self.best = float("inf")
        self.best_dep_val = [float("inf") for i in range(self.dim)]

        self.history_parts = []
        self.history_fitness_func = []

        self.history_best_dep_val = []
        self.history_best = []

    def run(self):
        for iteration in range(self.iterations):
            beta = np.random.uniform(0, 1)
            theta_max = np.random.uniform(0, 1)
            a = np.random.uniform(0, 1)

            prev_best = self.best
            self.best = min(self.best, self.fitness_func[np.argmin(self.fitness_func)])
            if prev_best != self.best:
                self.best_dep_val = self.bees[np.argmin(self.fitness_func)]

            self.history_best_dep_val.append(self.best_dep_val)
            self.history_best.append(self.best)

            self.bees = self.bees[np.argsort(self.fitness_func)]
            self.fitness_func = np.sort(self.fitness_func)

            # Bee worker phase
            for l in range(self.areas_num):
                Z = self.bees_depart_area_elite_num if l <= self.areas_elite_num else self.bees_depart_area_num
                theta = theta_max * a ** l
                x = np.zeros((Z, self.dim))
                for z in range(Z):
                    for j in range(self.dim):
                        x[z, j] = self.bees[z, j] + theta * beta * (self.limits[j][1] - self.limits[j][0]) * (-1+2*np.random.rand())
                        x[z, j] = max(self.limits[j][0], x[z, j])
                        x[z, j] = min(self.limits[j][1], x[z, j])
                        if j in self.integer:
                            x[z, j] = np.round(x[z, j])

                # for z in range(Z):
                #     x[z] = self.bees[z] + theta * beta * (np.array(self.limits)[:, 1] - np.array(self.limits)[:, 0]) * (-1+2*np.random.rand())
                # for d in range(self.dim):
                #     x[x > self.x_high[d]] = self.x_high[d]
                #     x[x < self.x_low[d]] = self.x_low[d]

                fitness_func = np.array([self.function(part) for part in x])
                z = np.argmin(fitness_func)

                if fitness_func[z] < self.function(self.bees[l]):
                    self.bees[l] = x[z]
                    self.fitness_func[l] = fitness_func[z]

            self.history_parts.append(self.bees)
            self.history_fitness_func.append(self.fitness_func)
            # Bee search phase
            self.bees = np.random.uniform(self.x_low, self.x_high, (self.pop_size, self.dim))
            self.fitness_func = np.array([self.function(part) for part in self.bees])

        self.plot(**self.kwargs)
        print(self.best, self.best_dep_val)
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


if __name__ == "__main__":
    # -------------------Rastrigin-------------------- #
    # def F(X):
    #     A = 10
    #     length = len(X)
    #     result = A*length
    #     for x in X:
    #         result += x**2-A*np.cos(2*np.pi*x)
    #     return result
    # bee = BEE(50, 100, 10, 1, 10, 1, F, [[-5.12, 5.12], [-5.12, 5.12]], d2=True, show=True).run()

    # ------------------Rozenbrock------------------- #
    # def F(X):
    #     # BonkBonk = 1e3
    #     x, y, = X
    #     f1 = (x-1)**3 - y + 1 < 0
    #     f2 = x + y - 2 < 0
    #     if f1 and f2:
    #         return (1 - x)**2 + 100*(y - x**2)**2
    #     else:
    #         return float('inf')
    #         # return (1 - x)**2 + 100*(y - x**2)**2 + BonkBonk

    # bee = BEE(50, 100, 25, 10, 8, 4, F, [[-1.5, 1.5], [-0.5, 2.5]], d2=True, show=True).run()

    # ----------------------Rozenbrock--------------------------- #
    # def F(X):
    #     # BonkBonk = 1e5
    #     x, y, = X
    #     f1 = x**2+y**2 < 2
    #     if f1:
    #         return (1 - x)**2 + 100*(y - x**2)**2
    #     else:
    #         return float('inf')

    # bee = BEE(50, 100, 25, 10, 8, 4, F, [[-1.5, 1.5], [-1.5, 1.5]], d2=True, show=True).run() # , save="Lab2/BEE.gif"

    # -------------------Mishri-Berda------------------------------ #
    # def F(X):
    #     x, y, = X
    #     f1 = (x+5)**2+(y+5)**2 < 25
    #     if f1:
    #         return np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    #     else:
    #         return float('inf')

    # bee = BEE(50, 100, 25, 10, 8, 4, F, [[-10, 0], [-6.5, 0]], d2=True, show=True).run()

    # ------------------Siminonesku------------------------------ #
    # def F(X):
    #     x, y, = X
    #     f1 = x**2+y**2 < (1 + 0.2*np.cos(8*np.arctan(x/y)))**2
    #     if f1:
    #         return 0.1*x*y
    #     else:
    #         return float('inf')

    # bee = BEE(50, 100, 25, 10, 8, 4, F, [[-1.25, 1.25], [-1.25, 1.25]], d2=True, show=True).run()

    # -----------------Reductor---------------------- #
    def F(X):
        x1, x2, x3, x4, x5, x6, x7, = X
        f1 = 27/(x1*x2**2*x3) - 1  < 0
        f2 = 397.5/(x1*x2**2*x3**2) - 1 < 0
        f3 = 1.93*x4**3/(x2*x3*x6**2) - 1 < 0
        f4 = 1.93/(x2*x3*x7**4) - 1 < 0
        f5 = 1.0/(110*x6**3) * np.sqrt(((745*x4)/(x2*x3))**2 + 16.9 * 10**6) - 1 < 0
        f6 = 1.0/(85*x7**3) * np.sqrt(((745*x5)/(x2*x3))**2 + 157.5 * 10**6) - 1 < 0
        f7 = x2*x3/40 - 1 <= 0
        f8 = 5*x2/x1 - 1 <= 0
        f9 = x1/(12*x2) - 1 <= 0
        f10 = (1.5*x6 + 1.9)/x4 - 1 <= 0
        f11 = (1.1*x7 + 1.9)/x5 - 1 <= 0
        if f1 and f2 and f3 and f4 and f5 and f6 and f7 and f8 and f9 and f10 and f11:
            return 0.7854*x1*x2**2*(3.3333*x3**2 + 14.9334*x3 - 43.0934) - 1.508*x1*(x6**2 + x7**2) + 7.4777*(x6**3 + x7**3) + 0.7854*(x4*x6**2 + x5*x7**2)
        else:
            return float('inf')

    bee = BEE(5000, 100, 25, 10, 2, 4, F, [[2.6, 3.6], [0.7, 0.8], [17, 28], [7.3, 8.3], [7.8, 8.3], [2.9, 3.9], [5.0, 5.5]], d2=True, show=False, integer=[2])
    result = bee.run()
    print(*result)
    print(F(result[1]))

    # -----------------Trail----------------------------- #
    # def F(X):
    #     x1, x2, x3, = X
    #     f1 = 1-(x2**3*x3)/(7.178*x1**4) <= 0
    #     f2 = (4*x2**2-x1*x2)/(12.566*(x2*x1**3) - x1**4) + 1/(5.108*x1**2) - 1 <= 0
    #     f3 = 1 - (140.45*x1)/(x2**2*x3) <= 0
    #     f4 = (x2+x1)/(1.5) - 1 <= 0
    #     if f1 and f2 and f3 and f4:
    #         return (x3 + 2)*x2*x1**2
    #     else:
    #         return float('inf')
    # bee = BEE(50, 100, 25, 10, 8, 4, F, [[0.005, 2.0], [0.25, 1.3], [2.0, 15.0]], d2=True, show=False, integer=[2])
    # result = bee.run()
    # print(*result)
    # print(F(result[1]))
