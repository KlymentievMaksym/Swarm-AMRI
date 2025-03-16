import numpy as np

if __name__ == "__main__":
    from Algoritm import Algorithm
else:
    from .Algoritm import Algorithm


class Cuckoo(Algorithm):
    def __init__(self, pop_size: int, iterations: int, prob_to_detect: float, eggs_to_move: int, eggs_to_detect: int, random_limits, function, limits, **kwargs):  # , eggs_to_detect: int = 1
        super().__init__(pop_size, iterations, random_limits, function, limits, **kwargs)

        self.prob_to_detect = prob_to_detect
        self.eggs_to_move = eggs_to_move
        self.eggs_to_detect = eggs_to_detect

    def run(self, **kwargs):
        self.run_before(**kwargs)
        for iteration in range(self.iterations):
            # CODE Start
            for _ in range(self.eggs_to_move):
                self.beta = np.random.uniform(self.low, self.high)
                k = np.random.randint(self.parts.shape[0])
                x_cur = self.parts[k] + self.beta * (self.x_high - self.x_low)*np.random.uniform(-1, 1, self.dim)
                x_cur = np.clip(x_cur, self.x_low, self.x_high)
                for i in self.integer:
                    x_cur[i] = np.round(x_cur[i])
                new_best = self.function(x_cur)
                if new_best < self.best:
                    self.best = new_best
                    self.best_dep_val = x_cur
                    self.parts[k] = x_cur
                    self.fitness_func[k] = new_best

                if np.random.rand() < self.prob_to_detect:
                    bad_eggs_index = np.argsort(self.fitness_func)[::-1][:self.eggs_to_detect]
                    for m in bad_eggs_index:
                    # m = np.argmax(self.fitness_func)
                        self.beta = np.random.uniform(self.low, self.high)
                        self.parts[m] = self.parts[m] + self.beta * (self.x_high - self.x_low)*np.random.uniform(-1, 1, self.dim)
                        self.parts[m] = np.clip(self.parts[m], self.x_low, self.x_high)
                        for i in self.integer:
                            self.parts[m, i] = np.round(self.parts[m, i])
                        self.fitness_func[m] = self.function(self.parts[m])
                        if self.fitness_func[m] < self.best:
                            self.best = self.fitness_func[m]
                            self.best_dep_val = self.parts[m]
            # CODE Finish

            self.check
            self.save
            self.progress_bar(iteration, self.iterations, name="Cuckoo")
            if self.same and self.break_faster:
                break
        return self.run_after


if __name__ == "__main__":
    def Rastrigin(X):
        A = 10
        length = len(X)
        result = A*length
        for x in X:
            result += x**2-A*np.cos(2*np.pi*x)
        return result

    Rastrigin_limits = [[-5.12, 5.12]]
    Cuckoo(500, 1000, 1, 20, 100, [.1, .4], Rastrigin, Rastrigin_limits*2, d2=True, plot=True, show=True, interval=.1, fps=20).run(save="Lab3/Images/Rastrigin.gif")  #, save="Lab3/Images/Rastrigin.gif"

    # def Mishri_Berda(X):
    #     x, y, = X
    #     f1 = (x+5)**2+(y+5)**2 < 25
    #     if f1:
    #         return np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    #     else:
    #         return float('inf')

    # Mishri_Berda_limits = [[-10, 0], [-6.5, 0]]
    # Cuckoo(100, 500, 1, [.1, 1], Mishri_Berda, Mishri_Berda_limits, d2=True, plot=True, show=True, fps=60).run()  #, save="Lab3/Images/Mishri_Berda.gif"
