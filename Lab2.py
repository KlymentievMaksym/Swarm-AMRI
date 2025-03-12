from Algorithms.Bee import BEE
from Algorithms.Firefly import FF
from Algorithms.PSO import PSO

import numpy as np
import numba as nb

import matplotlib.pyplot as plt


# --------------------------------------- #
# @nb.jit()
def Rastrigin(X):
    A = 10
    length = len(X)
    result = A*length
    for x in X:
        result += x**2-A*np.cos(2*np.pi*x)
    return result


Rastrigin_limits = [[-5.12, 5.12]]


# --------------------------------------- #
# @nb.jit()
def Rozenbrock1(X):
    x, y, = X
    f1 = (x-1)**3 - y + 1 < 0
    f2 = x + y - 2 < 0
    if f1 and f2:
        return (1 - x)**2 + 100*(y - x**2)**2
    else:
        return float('inf')


Rozenbrock1_limits = [[-1.5, 1.5], [-0.5, 2.5]]


# --------------------------------------- #
# @nb.jit()
def Rozenbrock2(X):
    x, y, = X
    f1 = x**2+y**2 < 2
    if f1:
        return (1 - x)**2 + 100*(y - x**2)**2
    else:
        return float('inf')


Rozenbrock2_limits = [[-1.5, 1.5], [-1.5, 1.5]]


# --------------------------------------- #
# @nb.jit()
def Mishri_Berda(X):
    x, y, = X
    f1 = (x+5)**2+(y+5)**2 < 25
    if f1:
        return np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    else:
        return float('inf')


Mishri_Berda_limits = [[-10, 0], [-6.5, 0]]


# --------------------------------------- #
# @nb.jit()
def Siminonesku(X):
    x, y, = X
    f1 = x**2+y**2 < (1 + 0.2*np.cos(8*np.arctan(x/y)))**2
    if f1:
        return 0.1*x*y
    else:
        return float('inf')


Siminonesku_limits = [[-1.25, 1.25], [-1.25, 1.25]]


# --------------------------------------- #
@nb.jit()
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


# --------------------------------------- #
# @nb.jit()
def Trail(X):
    x1, x2, x3, = X
    f1 = 1-(x2**3*x3)/(7.178*x1**4) <= 0
    f2 = (4*x2**2-x1*x2)/(12.566*(x2*x1**3) - x1**4) + 1/(5.108*x1**2) - 1 <= 0
    f3 = 1 - (140.45*x1)/(x2**2*x3) <= 0
    f4 = (x2+x1)/(1.5) - 1 <= 0
    if f1 and f2 and f3 and f4:
        return (x3 + 2)*x2*x1**2
    else:
        return float('inf')


Trail_limits = [[0.005, 2.0], [0.25, 1.3], [2.0, 15.0]]


# --------------------------------------- #
# def F(X):
#     x, y, = X
#     f1 = x**2+y**2 < 2
#     if f1:
#         return (1 - x)**2 + 100*(y - x**2)**2
#     else:
#         return float('inf')

# def Opt(X):
#     # print(X.astype(int))
#     return FF(*X.astype(int), [0, 1], F, [[-1.5, 1.5], [-1.5, 1.5]], progress=False).run()[0]
#     # return BEE(*X.astype(int), [0, 1], F, [[-1.5, 1.5], [-1.5, 1.5]]).run()[0]

# # bee = BEE(160, 44, 33, 10, 6, 2, [0, 1], F, [[-1.5, 1.5], [-1.5, 1.5]], d2=True, show=True, plot=True).run()
# # ff = FF(50, 50, [0, 1], F, [[-1.5, 1.5], [-1.5, 1.5]], d2=True, show=True).run()
# # pso = PSO(40, 70, [0, 4], [-.1, .1], Opt, [[50, 200], [10, 50], [10, 40], [5, 20], [2, 10], [1, 5]], d2=False, show=True, integer=[range(6)]).run()
# pso = PSO(10, 10, [0, 4], [-.1, .1], Opt, [[50, 100], [10, 50]], d2=False, show=True, integer=[range(2)]).run()
# print(*pso)

# --------------------------------------- #
# 2. (2 бали) Розробити програмне забезпечення (мова програмування і середовище розробки за вибором студента) для тестування описаних вище алгоритмів
# для знаходження глобального екстремуму функції Растринга (безумовна оптимізація, підрозділ 4.1 даної роботи).
# Взяти розмірність задачі n = 2, n = 5, n = 10, n = 15. Подати результати тестування у вигляді графіків, як у попередній роботі.
# Кількість запусків програмного застосунку) для 3-х алгоритмів будь-яке (на вибір студента).

# ns = [2, 5, 10, 15]
# for n in ns:
#     pso = []
#     for i in range(10):
#         pso.append(PSO(50, 50, [0, 4], [-.1, .1], Rastrigin, [[-1.5, 1.5]*n], history=True, break_faster=False).run()[0])
#     mean_pso = np.mean(pso, axis=0)
#     plt.plot(range(len(mean_pso)), mean_pso, label="PSO")
#     bee = []
#     for i in range(10):
#         bee.append(BEE(50, 50, 25, 13, 7, 4, [0, 1], Rastrigin, [[-1.5, 1.5]*n], history=True, break_faster=False).run()[0])
#     mean_bee = np.mean(bee, axis=0)
#     plt.plot(range(len(mean_bee)), mean_bee, label="BEE")
#     ff = []
#     for i in range(10):
#         ff.append(FF(50, 50, [0, 1], Rastrigin, [[-1.5, 1.5]*n], history=True, break_faster=False).run()[0])
#     mean_ff = np.mean(ff, axis=0)
#     plt.plot(range(len(mean_ff)), mean_ff, label="FF")

#     plt.legend()
#     plt.grid()
#     plt.savefig(f"Lab2/Images/Rastring_{n}.png")
#     # plt.show()
#     plt.close()

# --------------------------------------- #
# 3. (3 бали) Модифікувати програми так (для трьох алгоритмів), щоб можна було бачити процес пошуку глобального екстремуму функцій з обмеженнями, які наведені далі у 4-му розділі даної роботи.
# Взяти дві на вибір функції.
# Показати динаміку збіжності кожного з методів на поставлених задачах.

# funcs = [(Rozenbrock2, Rozenbrock2_limits), (Siminonesku, Siminonesku_limits)]
# for func, limit in funcs:
#     PSO(50, 50, [0, 4], [-.5, .5], func, limit, break_faster=True, d2=True, show=True, count=10).run()
#     BEE(50, 50, 25, 13, 7, 4, [0, 1], func, limit, break_faster=True, d2=True, show=True, count=10).run()
#     FF(50, 50, [0, 1], func, limit, break_faster=True, d2=True, show=True, count=10).run()

#     pso = []
#     for i in range(10):
#         pso.append(PSO(50, 50, [0, 4], [-.1, .1], func, limit, history=True, break_faster=False).run()[0])
#     mean_pso = np.mean(pso, axis=0)
#     plt.plot(range(len(mean_pso)), mean_pso, label="PSO")
#     bee = []
#     for i in range(10):
#         bee.append(BEE(50, 50, 25, 13, 7, 4, [0, 1], func, limit, history=True, break_faster=False).run()[0])
#     mean_bee = np.mean(bee, axis=0)
#     plt.plot(range(len(mean_bee)), mean_bee, label="BEE")
#     ff = []
#     for i in range(10):
#         ff.append(FF(50, 50, [0, 1], func, limit, history=True, break_faster=False).run()[0])
#     mean_ff = np.mean(ff, axis=0)
#     plt.plot(range(len(mean_ff)), mean_ff, label="FF")

#     plt.legend()
#     plt.grid()
#     plt.savefig(f"Lab2/Images/{func.__name__}.png")
#     # plt.show()
#     plt.close()

# --------------------------------------- #
# 4. (1 бал) Модифікувати програми для вирішення прикладного завдання (трьома алгоритмами), оптимізаційна математична модель якого наведена у підрозділі 5.1 даної роботи.
# Подати результати так само, як для завдання з попереднього пункту.

# funcs = [(Reductor, Reductor_limits)]
# for func, limit in funcs:

#     # PSO(100, 100, [0, 4], [-1.1, 1.1], func, limit, show=True, plot=False).run()
#     # BEE(100, 100, 25, 13, 7, 4, [0, 1], func, limit, show=True, plot=False).run()
#     # FF(100, 100, [0, 1], func, limit, show=True, plot=False).run()
#     # plt.legend()
#     # plt.grid()
#     # plt.savefig(f"Lab2/Images/{func.__name__}_one_time.png")
#     # # plt.show()
#     # plt.close()

#     pso = []
#     for i in range(10):
#         pso.append(PSO(100, 100, [0, 4], [-1.1, 1.1], func, limit, history=True, break_faster=False).run()[0])
#     mean_pso = np.mean(pso, axis=0)
#     plt.plot(range(len(mean_pso)), mean_pso, label="PSO")
#     bee = []
#     for i in range(10):
#         bee.append(BEE(100, 100, 25, 13, 7, 4, [0, 1], func, limit, history=True, break_faster=False).run()[0])
#     mean_bee = np.mean(bee, axis=0)
#     plt.plot(range(len(mean_bee)), mean_bee, label="BEE")
#     ff = []
#     for i in range(10):
#         ff.append(FF(100, 100, [0, 1], func, limit, history=True, break_faster=False).run()[0])
#     mean_ff = np.mean(ff, axis=0)
#     plt.plot(range(len(mean_ff)), mean_ff, label="FF")

#     plt.legend()
#     plt.grid()
#     plt.savefig(f"Lab2/Images/{func.__name__}.png")
#     # plt.show()
#     plt.close()

# --------------------------------------- #
# 5. (2 бали додаткові) Адаптувати програми для вирішення прикладного завдання (трьома алгоритмами), оптимізаційна математична модель якого наведена у підрозділі 5.2 даної роботи.
# Подати результати так само, як для завдання з попереднього пункту.

# funcs = [(Trail, Trail_limits)]
# for func, limit in funcs:
#     pso = []
#     for i in range(10):
#         pso.append(PSO(100, 100, [0, 4], [-.1, .1], func, limit, history=True, break_faster=False).run()[0])
#     mean_pso = np.mean(pso, axis=0)
#     plt.plot(range(len(mean_pso)), mean_pso, label="PSO")
#     bee = []
#     for i in range(10):
#         bee.append(BEE(100, 100, 25, 13, 7, 4, [0, 1], func, limit, history=True, break_faster=False).run()[0])
#     mean_bee = np.mean(bee, axis=0)
#     plt.plot(range(len(mean_bee)), mean_bee, label="BEE")
#     ff = []
#     for i in range(10):
#         ff.append(FF(100, 100, [0, 1], func, limit, history=True, break_faster=False).run()[0])
#     mean_ff = np.mean(ff, axis=0)
#     plt.plot(range(len(mean_ff)), mean_ff, label="FF")

#     plt.legend()
#     plt.grid()
#     plt.savefig(f"Lab2/Images/{func.__name__}.png")
#     # plt.show()
#     plt.close()
