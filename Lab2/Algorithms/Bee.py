import numpy as np

from Algoritm import Algorithm


class BEE(Algorithm):
    def __init__(self, pop_size, iterations, areas_num, areas_elite_num, bees_depart_area_num, bees_depart_area_elite_num, beta, theta_max, alpha, function, limits, **kwargs):
        super().__init__(pop_size, iterations, function, limits, **kwargs)

        self.areas_num = areas_num
        self.areas_elite_num = areas_elite_num
        self.bees_depart_area_num = bees_depart_area_num
        self.bees_depart_area_elite_num = bees_depart_area_elite_num

        self.beta = beta
        self.theta_max = theta_max
        self.alpha = alpha

    def run(self):
        for iteration in range(self.iterations):
            self.beta = np.random.uniform(0, 1)
            self.theta_max = np.random.uniform(0, 1)
            self.alpha = np.random.uniform(0, 1)

            prev_best = self.best
            self.best = min(self.best, self.fitness_func[np.argmin(self.fitness_func)])
            if prev_best != self.best:
                self.best_dep_val = self.parts[np.argmin(self.fitness_func)]

            self.history_best_dep_val.append(self.best_dep_val)
            self.history_best.append(self.best)

            self.parts = self.parts[np.argsort(self.fitness_func)]
            self.fitness_func = np.sort(self.fitness_func)

            # Bee worker phase
            for l in range(self.areas_num):
                Z = self.bees_depart_area_elite_num if l <= self.areas_elite_num else self.bees_depart_area_num
                theta = self.theta_max * self.alpha ** l
                x = np.zeros((Z, self.dim))
                # for z in range(Z):
                #     for j in range(self.dim):
                #         x[z, j] = self.parts[z, j] + theta * self.beta * (self.limits[j][1] - self.limits[j][0]) * (-1+2*np.random.rand())
                #         x[z, j] = max(self.limits[j][0], x[z, j])
                #         x[z, j] = min(self.limits[j][1], x[z, j])
                #         if j in self.integer:
                #             x[z, j] = np.round(x[z, j])

                for z in range(Z):
                    x[z] = self.parts[z] + theta * self.beta * (np.array(self.limits)[:, 1] - np.array(self.limits)[:, 0]) * (-1+2*np.random.rand(self.dim))
                    x[z] = np.maximum(self.limits[:, 0], np.minimum(self.limits[:, 1], x[z]))

                fitness_func = np.array([self.function(part) for part in x])
                z = np.argmin(fitness_func)

                if fitness_func[z] < self.function(self.parts[l]):
                    self.parts[l] = x[z]
                    self.fitness_func[l] = fitness_func[z]

            self.history_parts.append(self.parts)
            self.history_fitness_func.append(self.fitness_func)
            # Bee search phase
            self.parts[self.areas_num:] = np.random.uniform(self.x_low, self.x_high, (self.pop_size-self.areas_num, self.dim))
            self.fitness_func = np.array([self.function(part) for part in self.parts])

        self.plot(**self.kwargs)
        print(self.best, self.best_dep_val)
        return self.best, self.best_dep_val


if __name__ == "__main__":
    # -------------------Rastrigin-------------------- #
    # def F(X):
    #     A = 10
    #     length = len(X)
    #     result = A*length
    #     for x in X:
    #         result += x**2-A*np.cos(2*np.pi*x)
    #     return result
    # bee = BEE(50, 100, 20, 15, 10, 20, .4, .5, .9, F, [[-5.12, 5.12], [-5.12, 5.12]], d2=True, show=True).run()

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

    # bee = BEE(100, 50, 30, 15, 10, 5, .4, .5, .9, F, [[-1.5, 1.5], [-0.5, 2.5]], d2=True, show=True).run()

    # ----------------------Rozenbrock--------------------------- #
    def F(X):
        # BonkBonk = 1e5
        x, y, = X
        f1 = x**2+y**2 < 2
        if f1:
            return (1 - x)**2 + 100*(y - x**2)**2
        else:
            return float('inf')

    bee = BEE(100, 50, 50, 15, 8, 4, .4, .5, .7, F, [[-1.5, 1.5], [-1.5, 1.5]], d2=True, show=True).run() # , save="Lab2/BEE.gif"

    # -------------------Mishri-Berda------------------------------ #
    # def F(X):
    #     x, y, = X
    #     f1 = (x+5)**2+(y+5)**2 < 25
    #     if f1:
    #         return np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    #     else:
    #         return float('inf')

    # bee = BEE(100, 50, 50, 15, 8, 4, .4, .5, .7, F, [[-10, 0], [-6.5, 0]], d2=True, show=True).run()

    # ------------------Siminonesku------------------------------ #
    # def F(X):
    #     x, y, = X
    #     f1 = x**2+y**2 < (1 + 0.2*np.cos(8*np.arctan(x/y)))**2
    #     if f1:
    #         return 0.1*x*y
    #     else:
    #         return float('inf')

    # bee = BEE(100, 50, 50, 15, 8, 4, .4, .5, .7, F, [[-1.25, 1.25], [-1.25, 1.25]], d2=True, show=True).run()

    # -----------------Reductor---------------------- #
    # def F(X):
    #     x1, x2, x3, x4, x5, x6, x7, = X
    #     f1 = 27/(x1*x2**2*x3) - 1  < 0
    #     f2 = 397.5/(x1*x2**2*x3**2) - 1 < 0
    #     f3 = 1.93*x4**3/(x2*x3*x6**2) - 1 < 0
    #     f4 = 1.93/(x2*x3*x7**4) - 1 < 0
    #     f5 = 1.0/(110*x6**3) * np.sqrt(((745*x4)/(x2*x3))**2 + 16.9 * 10**6) - 1 < 0
    #     f6 = 1.0/(85*x7**3) * np.sqrt(((745*x5)/(x2*x3))**2 + 157.5 * 10**6) - 1 < 0
    #     f7 = x2*x3/40 - 1 <= 0
    #     f8 = 5*x2/x1 - 1 <= 0
    #     f9 = x1/(12*x2) - 1 <= 0
    #     f10 = (1.5*x6 + 1.9)/x4 - 1 <= 0
    #     f11 = (1.1*x7 + 1.9)/x5 - 1 <= 0
    #     if f1 and f2 and f3 and f4 and f5 and f6 and f7 and f8 and f9 and f10 and f11:
    #         return 0.7854*x1*x2**2*(3.3333*x3**2 + 14.9334*x3 - 43.0934) - 1.508*x1*(x6**2 + x7**2) + 7.4777*(x6**3 + x7**3) + 0.7854*(x4*x6**2 + x5*x7**2)
    #     else:
    #         return float('inf')

    # bee = BEE(10000, 100, 50, 15, 8, 4, .4, .5, .7, F, [[2.6, 3.6], [0.7, 0.8], [17, 28], [7.3, 8.3], [7.8, 8.3], [2.9, 3.9], [5.0, 5.5]], d2=False, show=False, integer=[2])
    # result = bee.run()
    # print(*result)
    # print(F(result[1]))

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
    # bee = BEE(100, 50, 50, 15, 8, 4, .4, .5, .7, F, [[0.005, 2.0], [0.25, 1.3], [2.0, 15.0]], d2=True, show=False, integer=[2])
    # result = bee.run()
    # print(*result)
    # print(F(result[1]))
