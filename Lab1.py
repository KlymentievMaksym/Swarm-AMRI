from Algorithms.Functions.Genetic import Genetic
from Algorithms.Functions.GrayWolf import GrayWolf

import numpy as np

import matplotlib.pyplot as plt

# --------------------------------------- #
# 3. (2 бали) Розробити програмне забезпечення (мова програмування і середовище розробки за вибором студента)
# для тестування описаних вище алгоритмів (генетичний алгоритм та алгоритм зграї сірих вовків)
# на одновимірній тестовій одноекстремальній функції.


def f_one_extremum(X):
    x = X
    # return x**2
    b = 10
    d = 10
    a = 0.5
    # return -1/(b*(a*x) + d)
    return 5 - 24*x + 17*x**2 - 11/3*x**3 + 1/4*x**4

# --------------------------------------- #
# 4. (1 бал) Модифікувати програми так, щоб можна було бачити процес пошуку глобального екстремуму.
# Тобто потрібно продемонструвати положення популяції на функції на кожній ітерації.


# Genetic(20, 40, 30, 8, f_one_extremum, [[0, 7]], d1=True, show=True).run() #save="Lab1/Images/Genetic.gif"
# GrayWolf(20, 40, [.1, 1], f_one_extremum, [[0, 7]], d1=True, show=True, save="Lab1/Images/GrayWolf.gif").run()


# --------------------------------------- #
# 5. (1 бал) Модифікувати програми на випадок пошуку глобального екстремуму двовимірної (багатовимірної) функції.
# Також показати процес пошуку глобального екстремуму.]
# Тобто потрібно продемонструвати положення популяції на функції на кожній ітерації.
# Показати зміну положення популяції на контурному графіку для двовимірних функцій.

# def Bohachevsky(X):
#     x, y = X
#     return x**2 + 2*y**2 - 0.3 * np.cos(3*np.pi*x) - 0.4 * np.cos(4*np.pi*y) + 0.7

# Genetic(50, 50, 50, 8, Bohachevsky, [[-2, 2] for _ in range(2)], d2=True, show=True, break_faster=False, save="Lab1/Images/Genetic_2.gif").run()
# GrayWolf(50, 50, [0.1, 1], Bohachevsky, [[-2, 2] for _ in range(2)], d2=True, show=True, break_faster=False, save="Lab1/Images/GrayWolf_2.gif").run()


# --------------------------------------- #
# 6. (2 бали) Для багатовимірних одноекстремальних функцій (на прикладі функції Растринга при n ≥ 3)
# показати лише графік пристосованості популяції (за критерієм значення функції кожного елемента популяції),
# а також показати графік відстані від кращого елемента популяції (значення функції) до відомого оптимального значення функції, яке вказане.
# Тобто в останньому в графіку по осі абсцис відкладається номер ітерації, а по осі ординат – найменше значення відстані між оптимальним значенням функції і кращим.

# def Rastrigin(X):
#     A = 10
#     length = len(X)
#     result = A*length
#     for x in X:
#         result += x**2-A*np.cos(2*np.pi*x)
#     return result


# Rastrigin_limits = [[-5.12, 5.12]]

# # gen = Genetic(50, 50, 50, 8, Rastrigin, Rastrigin_limits*3, d2=True, show=True, break_faster=False, save="Lab1/Images/Genetic_3_Rastrigin.gif", history=True).run()
# # wolf = GrayWolf(50, 50, [0.1, 1], Rastrigin, Rastrigin_limits*3, d2=True, show=True, break_faster=False, save="Lab1/Images/GrayWolf_3_Rastrigin.gif", history=True).run()

# gen = Genetic(500, 100, 500, 8, Rastrigin, Rastrigin_limits*4, d2=True, show=True, break_faster=False, history=True).run()[0]
# gen = 0 - np.array(gen)
# # print(gen)
# wolf = GrayWolf(500, 100, [0.1, 1], Rastrigin, Rastrigin_limits*4, d2=True, show=True, break_faster=False, history=True).run()[0]
# wolf = 0 - np.array(wolf)
# # print(wolf)
# plt.plot(range(len(gen)), gen, label="Genetic")
# plt.plot(range(len(wolf)), wolf, label="Wolf")

# plt.title("Distance from optimal for Rastring, n=4")
# plt.xlabel("Iteration")
# plt.ylabel("Distance from optimal")

# plt.grid()
# plt.legend()
# plt.savefig("Lab1/Images/Rastring4_distance.png")
# plt.show()

# --------------------------------------- #
# 7. (1 бал додатковий) Обрати будь-яку іншу з функцій, що не наведені у даній лабораторній роботі, знайти її глобальний екстремум, продемонструвати відповідні результати.

def Reductor(X):
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

Reductor_limits = [[2.6, 3.6], [0.7, 0.8], [17, 28], [7.3, 8.3], [7.8, 8.3], [2.9, 3.9], [5.0, 5.5]]

g = Genetic(1000, 300, 500, 16, Reductor, Reductor_limits, d2=True, show=True, break_faster=False, integer=[2]).run()
print(*g)
# d = GrayWolf(150, 100, [0.1, 1], Reductor, Reductor_limits, d2=True, show=True, break_faster=False, savep="Lab1/Images/GrayWolf_11.png", integer=[2]).run()
# # print(g, d)
