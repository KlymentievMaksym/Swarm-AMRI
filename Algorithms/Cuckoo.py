import numpy as np

if __name__ == "__main__":
    from Algoritm import Algorithm
else:
    from .Algoritm import Algorithm


class Cuckoo(Algorithm):
    def __init__(self, pop_size, iterations, random_limits, function, limits, **kwargs):
        super().__init__(pop_size, iterations, random_limits, function, limits, **kwargs)

    def run(self, **kwargs):
        self.run_before(**kwargs)
        for iteration in range(self.iterations):

            # CODE Start
            # CODE Finish

            self.check
            self.save
            self.progress_bar(iteration, self.iterations, name="BEE")
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
    Cuckoo(50, 50, [0, 1], Rastrigin, Rastrigin_limits, d2=True, show=True)
