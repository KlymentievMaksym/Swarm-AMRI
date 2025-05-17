import numba as nb

import numpy as np

if __name__ == "__main__":
    from Algorithms.Functions.Algoritm import Algorithm
else:
    from .Functions.Algoritm import Algorithm


class FF(Algorithm):
    def __init__(self, pop_size, iterations, random_limits, function, limits, **kwargs): # , beta_max, gamma, alpha
        super().__init__(pop_size, iterations, random_limits, function, limits, **kwargs)
        # self.beta_max = beta_max
        # self.gamma = gamma
        # self.alpha = alpha

    def run(self, **kwargs):
        # @nb.jit
        # def run_through(pop_size, parts, func, beta_max, gamma, alpha, limits, dim):
        #     for k in range(pop_size):
        #         for l in range(pop_size):
        #             if func(parts[l]) < func(parts[k]):
        #                 d = np.sqrt(sum([(y-x)**2 for x, y in zip(parts[k], parts[l])]))
        #                 beta = beta_max*np.exp(-gamma*d**2)
        #                 parts[k] = parts[k] + beta * (parts[l] - parts[k]) + alpha * (np.random.rand(dim)-0.5)
        #                 parts[k] = np.maximum(limits[:, 0], np.minimum(limits[:, 1], parts[k]))
        #     return parts
        # self.parts = run_through(self.pop_size, self.parts, self.function, self.beta_max, self.gamma, self.alpha, self.limits, self.dim)
        # firefly_indexes = np.where(self.fitness_func < self.function(self.parts[k]))[0]
        self.run_before(**kwargs)

        # self.beta_max = np.random.uniform(self.low, self.high)
        # self.gamma = np.random.uniform(self.low, self.high)
        # self.alpha = np.random.uniform(self.low, self.high)
        for iteration in range(self.iterations):
            self.parts = self.parts[np.argsort(self.fitness_func)]
            self.fitness_func = np.sort(self.fitness_func)
            # if iteration % 2 == 0:
            self.beta_max = np.random.uniform(self.low, self.high)
            self.gamma = np.random.uniform(self.low, self.high)
            self.alpha = np.random.uniform(self.low, self.high)
            for k in range(self.pop_size):
                for l in range(self.pop_size):
                    # self.beta_max = np.random.uniform(self.low, self.high)  
                    # self.gamma = np.random.uniform(self.low, self.high)
                    # self.alpha = np.random.uniform(self.low, self.high)
                    # if self.function(self.parts[l]) < self.function(self.parts[k]):
                    if self.fitness_func[l] < self.fitness_func[k]:
                        d = np.linalg.norm(self.parts[k] - self.parts[l])
                        # d = np.sqrt(np.sum([(y-x)**2 for x, y in zip(self.parts[k], self.parts[l])]))
                        beta = self.beta_max*np.exp(-self.gamma*d**2)
                        self.parts[k] = self.parts[k] + beta * (self.parts[l] - self.parts[k]) + self.alpha * (np.random.rand(self.dim)-0.5)
                        # self.parts[k] = np.clip(self.parts[k], self.limits[:, 0], self.limits[:, 1])
                        self.parts[k] = np.clip(self.parts[k], self.x_low, self.x_high)
                        for i in self.integer:
                            self.parts[k, i] = np.round(self.parts[k, i])
                        self.fitness_func[k] = self.function(self.parts[k])
            ks = np.argmin(self.fitness_func)
            new_best = self.function(self.parts[ks])
            if new_best < self.best:
                self.best = new_best
                self.best_dep_val = self.parts[ks]

            self.check
            self.save

            if iteration != 0 and iteration % 10 == 0:
                if (np.array(self.history_best[iteration-10:iteration]) == float("inf")).all():
                # if np.unique(np.array(self.history_best[iteration-10:iteration])).size == 1:
                    self.parts = np.random.uniform(self.x_low, self.x_high, (self.pop_size, self.dim))
                    for i in self.integer:
                        self.parts[:, i] = np.round(self.parts[:, i])
                    self.fitness_func = np.array([self.function(part) for part in self.parts])

            self.progress_bar(iteration, self.iterations, name="Firefly")
            if self.same and self.break_faster:
                break
        return self.run_after



if __name__ == "__main__":
    # -------------------Rastrigin-------------------- #
    # @nb.jit
    # def F(X):
    #     A = 10
    #     length = len(X)
    #     result = A*length
    #     for x in X:
    #         result += x**2-A*np.cos(2*np.pi*x)
    #     return result
    # ff = FF(50, 50, [0, 1], F, [[-5.12, 5.12], [-5.12, 5.12]], d2=True, show=True).run()

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

    # ff = FF(50, 50, [0, 2], F, [[-1.5, 1.5], [-0.5, 2.5]], d2=True, show=True).run()

    # ----------------------Rozenbrock--------------------------- #
    # def F(X):
    #     # BonkBonk = 1e5
    #     x, y, = X
    #     f1 = x**2+y**2 < 2
    #     if f1:
    #         return (1 - x)**2 + 100*(y - x**2)**2
    #     else:
    #         return float('inf')

    # ff = FF(50, 50, [0, .5], F, [[-1.5, 1.5], [-1.5, 1.5]], d2=True, show=True).run() # , save="Lab2/FF.gif"

    # -------------------Mishri-Berda------------------------------ #
    # def F(X):
    #     x, y, = X
    #     f1 = (x+5)**2+(y+5)**2 < 25
    #     if f1:
    #         return np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    #     else:
    #         return float('inf')

    # ff = FF(50, 50, [0, 1], F, [[-10, 0], [-6.5, 0]], d2=True, show=True).run()

    # ------------------Siminonesku------------------------------ #
    # def F(X):
    #     x, y, = X
    #     f1 = x**2+y**2 < (1 + 0.2*np.cos(8*np.arctan(x/y)))**2
    #     if f1:
    #         return 0.1*x*y
    #     else:
    #         return float('inf')

    # ff = FF(50, 50, [0, 1], F, [[-1.25, 1.25], [-1.25, 1.25]], d2=True, show=True).run()

    # -----------------Reductor---------------------- #
    @nb.jit
    def F(X):
        x1, x2, x3, x4, x5, x6, x7, = X
        f1 = 27 / (x1 * x2**2 * x3) - 1  <= 0
        f2 = 397.5 / (x1 * x2**2 * x3**2) - 1 <= 0
        f3 = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1 <= 0
        f4 = 1.93 / (x2 * x3 * x7**4) - 1 <= 0
        f5 = 1.0/(110 * x6**3) * np.sqrt(((745*x4) / (x2 * x3))**2 + 16.9 * 10**6) - 1 <= 0
        f6 = 1.0/(85 * x7**3) * np.sqrt(((745*x5) / (x2 * x3))**2 + 157.5 * 10**6) - 1 <= 0
        f7 = (x2*x3) / 40 - 1 <= 0
        f8 = 5*x2 / x1 - 1 <= 0
        f9 = x1 / (12 * x2) - 1 <= 0
        f10 = (1.5 * x6 + 1.9) / x4 - 1 <= 0
        f11 = (1.1 * x7 + 1.9) / x5 - 1 <= 0
        if f1 and f2 and f3 and f4 and f5 and f6 and f7 and f8 and f9 and f10 and f11:
            return 0.7854*x1*x2**2*(3.3333*x3**2 + 14.9334*x3 - 43.0934) - 1.508*x1*(x6**2 + x7**2) + 7.4777*(x6**3 + x7**3) + 0.7854*(x4*x6**2 + x5*x7**2)
        return float('inf')


    ff = FF(100, 500, [0.1, 1], F, [[2.6, 3.6], [0.7, 0.8], [17, 28], [7.3, 8.3], [7.8, 8.3], [2.9, 3.9], [5.0, 5.5]], d2=False, show=True, integer=[2], break_faster=True)
    result = ff.run()
    print(*result)

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

    # ff = FF(50, 50, [0, 1], F, [[0.005, 2.0], [0.25, 1.3], [2.0, 15.0]], d2=True, show=False, integer=[2])
    # result = ff.run()
    # print(*result)
    # print(F(result[1]))
