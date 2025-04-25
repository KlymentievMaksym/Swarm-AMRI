import numpy as np


if __name__ == "__main__":
    from Generator import Generator
else:
    from .Generator import Generator


# def E(x):
#     return x ** 2  # Функція, яку мінімізуємо


# def F(x):
#     return x + np.random.uniform(-1, 1)  # Невелике випадкове зміщення


# def T(i):
#     return 0.99 ** i  # Експоненціальне охолодження


# def Burnout(s1, t_min, t_max, E):
#     t = [t_max]
#     s = [s1]
#     i = 0
#     while t[-1] > t_min:
#         S_c = F(s[-1])
#         dE = E(S_c) - E(s[-1])
#         if dE <= 0:
#             s.append(S_c)
#         elif dE > 0:
#             if np.random.random() < np.exp(-dE / t[-1]):
#                 s.append(S_c)
#         i += 1
#         t.append(T(i))
#     return s[-1]


# if __name__ == "__main__":
#     print(f"Найкращий стан: {Burnout(1, 0.0001, 1, E):.5f}")


# import random
# import numpy as np

# def simulated_annealing(weights, values, capacity, initial_temp=1000, cooling_rate=0.99, stopping_temp=10, max_iter=1000):
#     n = len(weights)

#     # Генеруємо початкове рішення
#     current_solution = np.random.randint(0, 1, n)

#     def fitness(solution):
#         total_weight = sum(w * g for w, g in zip(weights, solution))
#         total_value = sum(v * g for v, g in zip(values, solution))
#         if total_weight > capacity:
#             return 0
#         return total_value

#     current_value = fitness(current_solution)
#     best_solution = current_solution.copy()
#     best_value = current_value

#     T = initial_temp
#     iteration = 0

#     while T > stopping_temp and iteration < max_iter:
#         # Сусід: зміна одного випадкового гена
#         neighbor = current_solution.copy()
#         idx = np.random.randint(0, n - 1)
#         neighbor[idx] = 1 - neighbor[idx]

#         neighbor_value = fitness(neighbor)

#         delta = neighbor_value - current_value

#         # Завжди приймаємо кращий варіант або гірший з певною ймовірністю
#         if delta <= 0:
#             current_solution = neighbor
#             current_value = neighbor_value

#             if current_value > best_value:
#                 best_solution = current_solution[:]
#                 best_value = current_value
#         elif delta > 0:
#             if np.random.random() < np.exp(delta / T):
#                 current_solution = neighbor
#                 current_value = neighbor_value

#                 if current_value > best_value:
#                     best_solution = current_solution[:]
#                     best_value = current_value

#         T *= cooling_rate
#         iteration += 1

#     return best_solution, best_value


# if __name__ == "__main__":
#     data, weight_max  = Generator(3).generate()
#     weight = data[:, 0]
#     value = data[:, 1]
#     print(weight, value, weight_max)
#     print(simulated_annealing(weight, value, weight_max))



import numpy as np

# Sample data
data, capacity  = Generator(7, min_amount=1).generate()
weights = data[:, 0]
values = data[:, 1]
# weights = np.array([2, 3, 6, 7, 5])
# values = np.array([6, 5, 8, 9, 6])
# capacity = 15
num_items = len(weights)

# Evaluation function
def fitness(solution):
    total_weight = np.sum(solution * weights)
    total_value = np.sum(solution * values)
    if total_weight > capacity:
        return 0  # Penalize overweight solutions
    return total_value

# Generate a random valid solution
def random_solution():
    while True:
        sol = np.random.randint(0, 2, size=num_items)
        if np.sum(sol * weights) <= capacity:
            return sol

# Generate a neighbor by flipping one bit
def get_neighbor(solution):
    neighbor = solution.copy()
    index = np.random.randint(0, num_items)
    neighbor[index] = 1 - neighbor[index]  # Flip 0 to 1 or 1 to 0
    if np.sum(neighbor * weights) <= capacity:
        return neighbor
    else:
        return solution  # Return original if neighbor is invalid

# Simulated Annealing function
def simulated_annealing(T=1000, T_min=1e-7, alpha=0.95, max_iter=1000):
    current_solution = random_solution()
    current_fitness = fitness(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness

    iteration = 0
    while T > T_min and iteration < max_iter:
        # for _ in range(max_iter):
        new_solution = get_neighbor(current_solution)
        new_fitness = fitness(new_solution)

        delta = new_fitness - current_fitness
        if delta <= 0:
            current_solution = new_solution
            current_fitness = new_fitness

            if current_fitness > best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_solution = new_solution
            current_fitness = new_fitness

            if current_fitness > best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness

        T *= alpha
        iteration += 1

    return best_solution, best_fitness

# Run the algorithm
solution, max_value = simulated_annealing()
print("Best solution:", solution)
print("Total value:", max_value)
print("Total weight:", np.sum(solution * weights))
print("Max weight:", capacity)
print("Possible weights:", weights)
print("Possible values:", values)
