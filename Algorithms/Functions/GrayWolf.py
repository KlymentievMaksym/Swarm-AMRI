import numpy as np

if __name__ == "__main__":
    from Algorithms.Functions.Algoritm import Algorithm
else:
    from .Functions.Algoritm import Algorithm


class GrayWolf(Algorithm):
    def __init__(self, pop_size, iterations, step_limits, function, limits, **kwargs):
        super().__init__(pop_size, iterations, [0, 0], function, limits, **kwargs)

        self.step_limits = step_limits
        self.steps = np.random.uniform(*self.step_limits, (self.pop_size, self.dim))

    def run(self, **kwargs):
        self.run_before(**kwargs)

        # print(self.parts)
        # print(self.parts != self.best_dep_val)
        # print(self.parts[self.parts != self.best_dep_val])
        # print(np.where(self.parts != self.best_dep_val, self.parts, np.nan))
        for iteration in range(self.iterations):
            self.parts = (self.parts + self.steps * (self.best_dep_val-self.parts)/np.linalg.norm(self.best_dep_val-self.parts))
            self.parts = np.clip(self.parts, self.x_low, self.x_high)
            for j in self.integer:
                self.parts[:, j] = np.round(self.parts[:, j])
            self.fitness_func = np.array([self.function(part) for part in self.parts])

            self.best = np.min(self.fitness_func)
            self.best_dep_val = self.parts[np.argmin(self.fitness_func)]

            self.check
            self.save

            self.progress_bar(iteration, self.iterations, name="GrayWolf")
            if self.same and self.break_faster:
                break
        return self.run_after



if __name__ == "__main__":
    def Bohachevsky(X):
        x, y = X
        return x**2 + 2*y**2 - 0.3 * np.cos(3*np.pi*x) - 0.4 * np.cos(4*np.pi*y) + 0.7

    GrayWolf(50, 50, [0.1, 1], Bohachevsky, [[-2, 2] for _ in range(2)], d2=True, show=True, break_faster=False).run()
