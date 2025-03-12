import numpy as np

if __name__ == "__main__":
    from Algoritm import Algorithm
else:
    from .Algoritm import Algorithm


class Genetic(Algorithm):
    def __init__(self, pop_size, iterations, child_size, bit_size, function, limits, **kwargs):
        super().__init__(pop_size, iterations, [0, 0], function, limits, **kwargs)

        self.child_size = child_size
        self.bit_size = bit_size

        self.parts = np.zeros((self.pop_size + self.child_size, self.dim))
        self.fitness_func = np.zeros((self.pop_size + self.child_size))

        self.parts_bits = np.random.randint(0, 2, [self.pop_size, self.bit_size, self.dim])
        self.parts_child_bits = np.random.randint(0, 2, [self.child_size, self.bit_size, self.dim])
        # child = np.append(self.parts_bits[0, :3], self.parts_bits[1, 3:], axis=0)
        # print(child)
        # print(child[:, 0])
        # print(np.append(self.parts_bits, self.parts_child_bits, axis=0).shape)

    def run(self, **kwargs):
        self.kwargs.update(kwargs)
        show = self.kwargs.get("show", False)
        save_location = self.kwargs.get("save", None)
        history = self.kwargs.get("history", False)
        break_faster = self.kwargs.get("break_faster", False)
        for iteration in range(self.iterations):
            # SELECTION
            for i in range(self.child_size):
                i1 = np.random.randint(self.pop_size)
                i2 = np.random.randint(self.pop_size)
                while i2 == i1 and self.pop_size != 1:
                    i2 = np.random.randint(self.pop_size)
                cross = np.random.randint(1, self.bit_size)
                if np.random.rand() < 0.5:
                    self.parts_child_bits[i, :] = np.append(self.parts_bits[i1, :cross], self.parts_bits[i2, cross:], axis=0)
                else:
                    self.parts_child_bits[i, :] = np.append(self.parts_bits[i2, :cross], self.parts_bits[i1, cross:], axis=0)

            # MUTATION
                rand_bit = np.random.randint(self.bit_size)
                self.parts_child_bits[i, rand_bit] = 1 - self.parts_child_bits[i, rand_bit]

            # ELLITISM
            self.parts_bits = np.append(self.parts_bits, self.parts_child_bits, axis=0)
            for i in range(self.pop_size + self.child_size):
                self.parts[i] = self.Dec(self.parts_bits[i, :])
                self.fitness_func[i] = self.function(self.parts[i])
            ind = np.argsort(self.fitness_func)
            self.fitness_func = np.sort(self.fitness_func)
            self.parts_bits = self.parts_bits[ind, :]
            self.parts = self.parts[ind]
            for j in self.integer:
                self.parts[:, j] = np.round(self.parts[:, j])
            self.parts_bits = self.parts_bits[:self.pop_size, :]

            self.best = self.fitness_func[0]
            self.best_dep_val = self.parts[0]

            if self.best != float("inf") and len(self.history_best) != 0:
                self.check_if_same(self.best, self.history_best[-1])

            if show or save_location is not None or history:
                show_all_population = self.kwargs.get("population", False)
                if show_all_population:
                    self.history_parts.append(self.parts.copy())
                    self.history_fitness_func.append(self.fitness_func.copy())
                else:
                    self.history_parts.append(self.parts[:self.pop_size].copy())
                    self.history_fitness_func.append(self.fitness_func[:self.pop_size].copy())

                self.history_best.append(self.best)
                self.history_best_dep_val.append(self.best_dep_val.copy())

            self.progress_bar(iteration, self.iterations, name="Genetic")
            if self.same and break_faster:
                break
        self.plot(**self.kwargs)
        if history:
            return self.history_best, self.history_best_dep_val, self.history_fitness_func, self.history_parts
        return self.best, self.best_dep_val

    def Dec(self, parts_bits_i):
        return self.bi2de(parts_bits_i)/(2**self.bit_size-1)*(self.x_high-self.x_low) + self.x_low

    def bi2de(self, binary_array):
        # print(binary_array)
        array_to_return = np.zeros(self.dim)
        for i in range(self.dim):
            array_to_return[i] = int("".join(map(str, binary_array[:, i])), 2)
        return array_to_return


# if __name__ == "__main__":
#     def F(X):
#         return sum(X)**2
#     def min_f(X):
#         x = X
#         return 5 - 24*x + 17*x**2 - 11/3*x**3 + 1/4*x**4
#     Genetic(10, 50, 20, 8, min_f, [[0, 7] for _ in range(1)], d1=True, show=True).run()
    
    # def Bohachevsky(X):
    #     x, y = X
    #     return x**2 + 2*y**2 - 0.3 * np.cos(3*np.pi*x) - 0.4 * np.cos(4*np.pi*y) + 0.7
    # Genetic(10, 50, 20, 8, Bohachevsky, [[-2, 2] for _ in range(2)], d2=True, show=True, break_faster=False, population=True).run()
