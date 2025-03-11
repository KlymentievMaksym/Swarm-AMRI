import numpy as np

if __name__ == "__main__":
    from Algoritm import Algorithm
else:
    from .Algoritm import Algorithm


class PSO(Algorithm):
    def __init__(self, pop_size, iterations, random_limits, limits_speed, function, limits, **kwargs):
        super().__init__(pop_size, iterations, random_limits, function, limits, **kwargs)

        self.limits_speed = limits_speed
        self.speed = np.random.uniform(self.limits_speed[0], self.limits_speed[1], (self.pop_size, self.dim))

        self.best_personal = [float("inf") for part in range(self.pop_size)]
        self.best_personal_dep_val = [[0 for i in range(self.dim)] for j in range(self.pop_size)]

    def run(self, **kwargs):
        self.kwargs.update(kwargs)
        show = self.kwargs.get("show", False)
        save_location = self.kwargs.get("save", None)
        for iteration in range(self.iterations):
            self.progress_bar(iteration, self.iterations, name="PSO")
            if show or save_location is not None:
                self.history_parts.append(self.parts.copy())

            self.fitnes_func = []
            for i in range(self.pop_size):
                # low_un = 0.000000001
                a1 = np.random.uniform(self.low, self.high)
                a2 = np.random.uniform(self.low, self.high)
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                # r1 = np.random.uniform(0, 1, self.dim)
                # r2 = np.random.uniform(0, 1, self.dim)

                self.fitnes_func.append(self.function(self.parts[i]))

                prev_best_personal = self.best_personal[i]
                self.best_personal[i] = min(self.best_personal[i], self.fitnes_func[i])
                if prev_best_personal != self.best_personal[i]:
                    self.best_personal_dep_val[i] = self.parts[i].copy()

                prev_best = self.best
                self.best = min(self.best, self.fitnes_func[i])
                if prev_best != self.best:
                    self.check_if_same(prev_best, self.best)
                    self.best_dep_val = self.parts[i].copy()

                to_best_self = (self.best_personal_dep_val[i] - self.parts[i]) * r1
                to_best_overal = (self.best_dep_val - self.parts[i]) * r2
                self.speed[i] = self.speed[i] + a1 * to_best_self + a2 * to_best_overal

                self.speed[i][self.speed[i] > self.limits_speed[1]] = self.limits_speed[1]
                self.speed[i][self.speed[i] < self.limits_speed[0]] = self.limits_speed[0]

                self.parts[i] = self.parts[i] + self.speed[i]
                # new_parts = np.clip(self.parts[i], self.x_low, self.x_high)
                # self.speed[i][new_parts != self.parts[i]] = -self.speed[i][new_parts != self.parts[i]]
                # self.parts[i] = new_parts
                for l in range(self.dim):
                    if self.parts[i][l] > self.x_high[l]:
                        self.parts[i][l] = self.x_high[l] - abs(self.parts[i][l] - self.x_high[l])
                        self.speed[i][l] = -self.speed[i][l]
                    elif self.parts[i][l] < self.x_low[l]:
                        self.parts[i][l] = self.x_low[l] + abs(self.parts[i][l] - self.x_low[l])
                        self.speed[i][l] = -self.speed[i][l]
                for inte in self.integer:
                    self.parts[i, inte] = np.round(self.parts[i, inte])
            if self.best != float("inf") and len(self.history_best) != 0:
                self.check_if_same(self.best, self.history_best[-1])

            if show or save_location is not None:
                self.history_fitness_func.append(self.fitnes_func.copy())
                self.history_best_dep_val.append(self.best_dep_val.copy())
                self.history_best.append(self.best)
            if self.same:
                break
        if show or save_location is not None:
            self.plot(**self.kwargs)
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
    # pso = PSO(40, 70, [0, 4], [-.1, .1], F, [[-5.12, 5.12], [-5.12, 5.12]], d2=True, show=True).run()

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

    # pso = PSO(40, 70, [0, 4], [-.1, .1], F, [[-1.5, 1.5], [-0.5, 2.5]], d2=True, show=True).run()

    # ----------------------Rozenbrock--------------------------- #
    # def F(X):
    #     # BonkBonk = 1e5
    #     x, y, = X
    #     f1 = x**2+y**2 < 2
    #     if f1:
    #         return (1 - x)**2 + 100*(y - x**2)**2
    #     else:
    #         return float('inf')

    # pso = PSO(40, 70, [0, 4], [-.1, .1], F, [[-1.5, 1.5], [-1.5, 1.5]], d2=True, show=True).run() # , save="Lab2/PSO.gif"

    # -------------------Mishri-Berda------------------------------ #
    # def F(X):
    #     x, y, = X
    #     f1 = (x+5)**2+(y+5)**2 < 25
    #     if f1:
    #         return np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    #     else:
    #         return float('inf')

    # pso = PSO(40, 70, [0, 4], [-.1, .1], F, [[-10, 0], [-6.5, 0]], d3=True, show=True).run()

    # ------------------Siminonesku------------------------------ #
    # def F(X):
    #     x, y, = X
    #     f1 = x**2+y**2 < (1 + 0.2*np.cos(8*np.arctan(x/y)))**2
    #     if f1:
    #         return 0.1*x*y
    #     else:
    #         return float('inf')

    # pso = PSO(40, 70, [0, 4], [-.1, .1], F, [[-1.25, 1.25], [-1.25, 1.25]], d2=True, show=True).run()

    # -----------------Reductor---------------------- #
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

    pso = PSO(100, 500, [0, 4], [-1.1, 1.1], F, [[2.6, 3.6], [0.7, 0.8], [17, 28], [7.3, 8.3], [7.8, 8.3], [2.9, 3.9], [5.0, 5.5]], d2=False, show=True, integer=[2])
    result = pso.run()
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
    # pso = PSO(100, 50, [0, 4], [-1.1, 1.1], F, [[0.005, 2.0], [0.25, 1.3], [2.0, 15.0]], d2=True, show=False, integer=[2])
    # result = pso.run()
    # print(*result)
    # print(F(result[1]))
