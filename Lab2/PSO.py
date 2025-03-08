import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PSO:
    def __init__(self, pop_size, iterations, limits_speed, function, limits, **kwargs):
        self.iterations = iterations
        self.pop_size = pop_size
        self.limits_speed = limits_speed
        self.function = function
        self.limits = limits
        
        self.dim = len(self.limits)

        self.x_low = [limit[0] for limit in self.limits]
        self.x_high = [limit[1] for limit in self.limits]
        self.parts = np.random.uniform(self.x_low, self.x_high, (self.pop_size, self.dim))
        dots = 100
        self.projection_dep_val = np.linspace(self.x_low, self.x_high, dots)
        # print((self.projection_dep_val))
        space = np.array([self.projection_dep_val[:, i] for i in range(self.dim)])
        space = np.meshgrid(*space)
        # print(space[0], space[0].shape, space[1].shape, sep="\n\n\n\n\n")
        self.projection = np.array([[self.function([space[i][j, k] for i in range(self.dim)]) for j in range(dots)] for k in range(dots)])
        # print(self.projection, self.projection.shape, sep='\n')
        # a = np.meshgrid(*[self.projection_dep_val[:, i] for i in range(self.dim)])
        # print(a)
        # self.projection = np.array([self.function(self.projection_dep_val[i]) for i in range(dots)])
        # self.projection = np.array([[self.function(space[i, j]) for i in range(dots)] for j in range(dots)])
        # print(self.projection)
        
        self.best_personal = [float("inf") for part in range(self.pop_size)]
        self.best_personal_dep_val = [[0 for i in range(self.dim)] for j in range(self.pop_size)]

        self.best = float("inf")
        self.best_dep_val = [0 for i in range(self.dim)]
        
        self.speed = np.random.uniform(0, 1, (self.pop_size, self.dim))
        # self.speed = self.speed/1000000000000000

        self.history_parts = []
        self.history_fitness_func = []

        self.history_best_dep_val = []
        self.history_best = []

        for iteration in range(self.iterations):
            self.history_parts.append(self.parts.copy())
            
            self.fitnes_func = []
            for i in range(self.pop_size):
            # for i in range(1):
                low_un = 0.000000001
                a1 = np.random.uniform(low_un, 4)
                a2 = np.random.uniform(low_un, 4)
                # r1 = np.random.rand()
                # r2 = np.random.rand()
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)

                self.fitnes_func.append(self.function(self.parts[i]))

                prev_best_personal = self.best_personal[i]
                self.best_personal[i] = min(self.best_personal[i], self.fitnes_func[i])
                if prev_best_personal != self.best_personal[i]:
                    self.best_personal_dep_val[i] = self.parts[i].copy()

                prev_best = self.best
                self.best = min(self.best, self.fitnes_func[i])
                if prev_best != self.best:
                    self.best_dep_val = self.parts[i].copy()

                # print(self.speed)
                to_best_self = (self.best_personal_dep_val[i] - self.parts[i]) * r1
                to_best_overal = (self.best_dep_val - self.parts[i]) * r2
                self.speed[i] = self.speed[i] + a1 * to_best_self + a2 * to_best_overal
                self.speed[i][self.speed[i] > self.limits_speed[1]] = self.limits_speed[1]
                self.speed[i][self.speed[i] < self.limits_speed[0]] = self.limits_speed[0]
                # print(self.speed)

                self.parts[i] = self.parts[i] + self.speed[i]
                for l in range(self.dim):
                    if self.parts[i][l] > self.x_high[l]:
                        self.parts[i][l] = self.x_high[l]
                    elif self.parts[i][l] < self.x_low[l]:
                        self.parts[i][l] = self.x_low[l]

            self.history_fitness_func.append(self.fitnes_func)

            self.history_best_dep_val.append(self.best_dep_val.copy())
            self.history_best.append(self.best)

        d3 = kwargs.get("d3", False)
        d2 = kwargs.get("d2", False)
        if d2:
            fig, ax1 = plt.subplots()
        elif d3:
            fig = plt.figure(figsize=plt.figaspect(2.))
            ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        cs = ax1.contourf(*space, self.projection, cmap="cool")
        fig.colorbar(cs)

        fig.canvas.manager.window.state('zoomed')
        # print(self.history_best)
        def update(frame):
            ax1.clear()
            ax1.set_title(f"Best solution: {self.history_best[frame]:.5f} | Iter {frame}")
            if d2:
                ax1.contourf(*space, self.projection, cmap="cool")
                ax1.scatter(*[self.history_parts[frame][:, i] for i in range(self.dim)], label="Population", c='white')
                ax1.scatter(*[self.history_best_dep_val[frame][i] for i in range(self.dim)], label="Best", c='yellow')
            elif d3:
                ax1.plot_surface(*space, self.projection, cmap="cool", alpha=0.8)
                ax1.scatter(*[self.history_parts[frame][:, i] for i in range(self.dim)], self.history_fitness_func[frame], label="Population", c='Black')
                ax1.scatter(*[self.history_best_dep_val[frame][i] for i in range(self.dim)], self.history_best[frame], label="Best", c='Red')
            ax1.legend()

        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.iterations, interval=10)
        # ani.save("1.gif")
        plt.show()
        
        
        # print(self.fitnes_func)


        


if __name__ == "__main__":
    def F(X):
        A = 10
        length = len(X)
        result = A*length
        for x in X:
            result += x**2-A*np.cos(2*np.pi*x)
        return result

    pso = PSO(50, 100, [-1, 1], F, [[-5.12, 5.12], [-5.12, 5.12]], d3=True)
