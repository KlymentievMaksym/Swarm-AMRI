import numpy as np

if __name__ == "__main__":
    from Algoritm import Algorithm
else:
    from .Algoritm import Algorithm


class Bat(Algorithm):
    def __init__(self, pop_size, iterations, r0, A0, alpha, beta, gamma, frequency_limits, random_limits, function, limits, **kwargs):
        super().__init__(pop_size, iterations, random_limits, function, limits, **kwargs)
        self.r0 = r0
        self.A0 = A0
        # self.r0 = [r0 for _ in range(self.pop_size)]
        # self.A0 = [A0 for _ in range(self.pop_size)]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.frequency_low, self.frequency_high = frequency_limits
        if self.frequency_low > self.frequency_high:
            self.frequency_high, self.frequency_low = self.frequency_low, self.frequency_high

    def run(self, **kwargs):
        self.run_before(**kwargs)

        # CODE Start

        self.speed = np.zeros((self.pop_size, self.dim))
        self.frequency = np.array([self.frequency_low for _ in range(self.pop_size)])

        for iteration in range(self.iterations):
            if self.random:
                self.alpha = np.random.uniform(self.low, self.high)
                self.beta = np.random.uniform(self.low, self.high)
                self.gamma = np.random.uniform(self.low, self.high)

                self.frequency_low = np.random.uniform(self.low, self.high)
                self.frequency_high = np.random.uniform(self.low+.5, self.high+.5)
                if self.frequency_low > self.frequency_high:
                    self.frequency_high, self.frequency_low = self.frequency_low, self.frequency_high

            for bat_index in range(self.pop_size):
                self.frequency = np.random.uniform(self.frequency_low, self.frequency_high)
                self.speed[bat_index] = self.speed[bat_index] + (self.best_dep_val - self.parts[bat_index])*self.frequency
                self.parts[bat_index] = self.parts[bat_index] + self.speed[bat_index]
                self.parts[bat_index] = np.clip(self.parts[bat_index], self.x_low, self.x_high)

                # self.fitness_func[bat_index] = self.function(self.parts[bat_index])
                # new_best = self.fitness_func[bat_index]
                # if new_best < self.best:
                #     self.best = new_best
                #     self.best_dep_val = self.parts[bat_index].copy()

                if np.random.rand() < self.r0:
                # if np.random.rand() < self.r0[bat_index]:
                    x_cur = self.best_dep_val + self.beta * (self.x_high - self.x_low) * np.random.uniform(-1, 1, self.dim)
                    x_cur = np.clip(x_cur, self.x_low, self.x_high)
                    new_best = self.function(x_cur)
                    # if new_best <= self.fitness_func[bat_index] and np.random.rand() < self.A0[bat_index]:
                    if new_best <= self.fitness_func[bat_index] and np.random.rand() < self.A0:
                        self.parts[bat_index] = x_cur.copy()
                        self.fitness_func[bat_index] = new_best
                        # self.A0[bat_index] *= self.alpha
                        # self.r0[bat_index] *= (1-np.exp(-1*self.gamma*(iteration+1)))
                        self.A0 *= self.alpha**(iteration/(self.dim*self.iterations))
                        self.r0 *= (1-np.exp(-1*self.gamma*(iteration+1)))
                    if new_best <= self.best:
                        self.best = new_best
                        self.best_dep_val = x_cur.copy()

                self.fitness_func[bat_index] = self.function(self.parts[bat_index])
                new_best = self.fitness_func[bat_index]
                if new_best < self.best:
                    self.best = new_best
                    self.best_dep_val = self.parts[bat_index].copy()
            # CODE Finish

            self.check
            self.save
            self.progress_bar(iteration, self.iterations, name="Bat")
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
    Bat(100, 500, .5, 1.5, .5, .5, .5, [.1, .5], [.1, .5], Rastrigin, Rastrigin_limits*2, history=False, d3=True, show=True, random=False, fps=20).run()  # save="Lab3/Images/Rastrigin_bat6.gif"


    # def Mishri_Berda(X):
    #     x, y, = X
    #     f1 = (x+5)**2+(y+5)**2 < 25
    #     if f1:
    #         return np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    #     else:
    #         return float('inf')

    # Mishri_Berda_limits = [[-10, 0], [-6.5, 0]]
    # Bat(50, 50, .5, 1.9, .1, .5, .5, [.1, .3], [.1, 1], Mishri_Berda, Mishri_Berda_limits, d2=True, plot=True, show=True, fps=60, random=False).run()  #, save="Lab3/Images/Mishri_Berda.gif"
