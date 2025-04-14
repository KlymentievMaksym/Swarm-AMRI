from Algorithms.Functions.Bat import Bat
from Algorithms.Cuckoo import Cuckoo

import numpy as np
import numba as nb

import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# 2. (3 бали) Розробити програмне забезпечення (мова програмування і середовище
# розробки за вибором студента) для тестування знаходження глобального
# екстремуму функцій з четвертого розділу роботи (для функції Растринга
# взяти n = 20). Протестувати кожен з розглянутих методів на даних функціях.
# Подати результати тестування у вигляді графіків, як у попередніх роботах.
# Кількість запусків програмного застосунку) для 3-х алгоритмів будь-яке (на
# вибір студента).

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
def Mishri_Berda(X):
    x, y, = X
    f1 = (x+5)**2+(y+5)**2 < 25
    if f1:
        return np.exp((1-np.cos(x))**2)*np.sin(y) + np.exp((1-np.sin(y))**2)*np.cos(x) + (x-y)**2
    else:
        return float('inf')


Mishri_Berda_limits = [[-10, 0], [-6.5, 0]]

# --------------------------------------- #

# ite = 3
# iterat = 10000
# iteratC = 10000
# pop = 100
# bat = []
# for i in range(ite):
#     bat.append(Bat(pop, iterat, .9, 1.9, .9, .5, .9, [.1, .3], [0, 1], Rastrigin, Rastrigin_limits*20, history=True, d2=True, show=False, random=True, fps=20).run()[0])  #, save="Lab3/Images/Rastrigin.gif"
# plt.plot(range(iterat), np.mean(bat, axis=0), label="Bat")
# cuckoo = []
# for i in range(ite):
#     cuckoo.append(Cuckoo(pop, iteratC, .5, 1, 1, [0, 1], Rastrigin, Rastrigin_limits*20, history=True, d2=True, show=False, random=True, fps=20).run()[0])  #, save="Lab3/Images/Rastrigin.gif"
# plt.plot(range(iteratC), np.mean(cuckoo, axis=0), label="Cuckoo")
# plt.legend()
# plt.grid()
# plt.savefig(f"Lab3/Images/{Rastrigin.__name__}.png")
# plt.show()

# bat = []
# for i in range(ite):
#     bat.append(Bat(pop, iterat, .9, 1.9, .9, .5, .9, [.1, .3], [0, 1], Mishri_Berda, Mishri_Berda_limits, history=True, d2=True, show=False, random=True, fps=20).run()[0])  #, save="Lab3/Images/Rastrigin.gif"
# plt.plot(range(iterat), np.mean(bat, axis=0), label="Bat")
# cuckoo = []
# for i in range(ite):
#     cuckoo.append(Cuckoo(pop, iteratC, .5, 1, 1, [0, 1], Mishri_Berda, Mishri_Berda_limits, history=True, d2=True, show=False, random=True, fps=20).run()[0])  #, save="Lab3/Images/Rastrigin.gif"
# plt.plot(range(iteratC), np.mean(cuckoo, axis=0), label="Cuckoo")
# plt.legend()
# plt.grid()
# plt.savefig(f"Lab3/Images/{Mishri_Berda.__name__}.png")
# plt.show()

# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# 3. (3 бали) Модифікувати відповідні програми для знаходження розв’язку будь-
# якої задачі з підрозділу 5.1. Показати динаміку збіжності обраного методу на
# обраній задачі, навести графічно відповідний розв’язок задач та числові
# параметри, які демонструють достатню збіжність методу на задачі. У якості
# цільової функції взяти:
#     a) f(α) = |x(t = b; α) − B|;
#     b) f(α) = (x(t = b; α) − B)**2,
#       де b – значення аргументу на границі, B – потрібне значення функції
#       на границі t = b для деякого значення α.
# Оцінити графічно, наскільки ефективно працюють методи для різних
# цільових функцій.


# --------------------------------------- #
def F511_plot(X):
    N = 101
    xa, xb = 1, 3
    h = (xb - xa) / (N - 1)
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]
    al = np.arange(-3, 3, 0.2)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    f = np.zeros_like(al)

    # x'(1) = 1, x(3) = 2, y(1) = 1, t = [1, 3]
    for l in range(len(al)):
        x[0] = al[l]
        y[0] = 1
        z[0] = 1
        for k in range(N - 1):
            # x' = z
            x_s = z[k]
            # y' = y**4 + x**3 - 3 * np.sin(t * x')
            y_s = y[k]**4 + x[k]**3 - 3 * np.sin(t[k] * x_s)
            # x'' + y'**2 * np.cos(x') = x**2 * t**2    
            # x'' = x**2 * t**2 - y'**2 * np.cos(x')
            # z' = x**2 * t**2 - y'**2 * np.cos(z)
            z_s = x[k]**2 * t[k]**2 - y_s**2 * np.cos(z[k])

            y[k+1] = y[k] + h * y_s
            x[k+1] = x[k] + h * x_s
            z[k+1] = z[k] + h * z_s
        # plt.plot(t, y)
        f[l] = (x[N-1] - 2) ** 2
    plt.plot(al, f)
    plt.grid()
    plt.show()


F511_limits = [[-1.5, .3]]

# F511_plot(10)


# --------------------------------------- #
def F512_plot(X):
    N = 101
    xa, xb = 1, 3
    h = (xb - xa) / (N - 1)
    t = np.linspace(xa, xb, N)
    al = np.arange(-3, 3, 0.2)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    f = np.zeros_like(al)

    # x(1) = 2, y(1) = 2, y'(3) = 0, t = [1, 3]

    for l in range(len(al)):
        x[0] = 2
        y[0] = 2
        z[0] = al[l]
        for k in range(N - 1):
            # y' = z
            y_s = z[k]
            # x' - y'**2 * np.cos(2*t + x) = 5*x**2 - 25*t**2
            x_s = y[k]**2 * np.cos(2*t[k] + x[k]) + 5*x[k]**2 - 25*t[k]**2
            # y'' = 4/(1 + y**2 * x**2) + 4 * np.sin(t * x')
            z_s = 4 / (1 + y[k]**2 * x[k]**2) + 4 * np.sin(t[k] * z[k])

            y[k+1] = y[k] + h * y_s
            x[k+1] = x[k] + h * x_s
            z[k+1] = z[k] + h * z_s
        # plt.plot(t, y)
        f[l] = (z[N-1] - 0) ** 2
    plt.plot(al, f)
    plt.grid()
    plt.show()


F512_limits = [[-3, 3]]
# F512_plot(10)


# --------------------------------------- #
def F513_plot(X):
    N = 101
    xa, xb = 1, 3
    h = (xb - xa) / (N - 1)
    t = np.linspace(xa, xb, N)
    al = np.arange(-3, 3, 0.2)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    f = np.zeros_like(al)

    for l in range(len(al)):
        # x(1) = 2, y(1) = 1, y'(3) = -1, t = [1, 3]
        x[0] = 2
        y[0] = 1
        z[0] = al[l]
        for k in range(N - 1):
            # y' = z
            y_s = z[k]

            # x' = 2*x**2 - 25*t**2 - np.sin(x * y * t)
            x_s = 2*x[k]**2 - 25*t[k]**2 - np.sin(x[k] * y[k] * t[k])

            # y'' = 1 - 4 * np.cos(x' * t)
            z_s = 1 - 4 * np.cos(x_s * t[k])

            y[k+1] = y[k] + h * y_s
            x[k+1] = x[k] + h * x_s
            z[k+1] = z[k] + h * z_s
        f[l] = (z[N-1] - -1) ** 2
    plt.plot(al, f)
    plt.grid()
    plt.show()


F513_limits = [[-3, 3]]
# F513_plot(10)


# --------------------------------------- #
def F511_square(X):
    N = 1001
    xa, xb = 1, 3
    # h = (xb - xa) / (N - 1)
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    # x'(1) = 1, x(3) = 2, y(1) = 1, t = [1, 3]
    x[0] = X
    y[0] = 1
    z[0] = 1
    for k in range(N - 1):
        # x' = z
        x_s = z[k]
        # y' = y**4 + x**3 - 3 * np.sin(t * x')
        y_s = y[k]**4 + x[k]**3 - 3 * np.sin(t[k] * x_s)
        # x'' + y'**2 * np.cos(x') = x**2 * t**2    
        # x'' = x**2 * t**2 - y'**2 * np.cos(x')
        # z' = x**2 * t**2 - y'**2 * np.cos(z)
        z_s = x[k]**2 * t[k]**2 - y_s**2 * np.cos(z[k])

        y[k+1] = y[k] + h * y_s
        x[k+1] = x[k] + h * x_s
        z[k+1] = z[k] + h * z_s
    return abs(x[N-1] - 2)


# Cuckoo(100, 100, .5, 1, 1, [0, 1], F511_square, F511_limits, d1=True, show=True).run(save=f"Lab3/Images/{F511_square.__name__}_Cuckoo.gif")
# Bat(100, 100, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F511_square, F511_limits, d1=True, show=True, random=False).run(save=f"Lab3/Images/{F511_square.__name__}_Bat.gif")


# --------------------------------------- #
def F512_square(X):
    N = 101
    xa, xb = 1, 3
    # h = (xb - xa) / (N - 1)
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    # x(1) = 2, y(1) = 2, y'(3) = 0, t = [1, 3]
    x[0] = 2
    y[0] = 2
    z[0] = X
    for k in range(N - 1):
        # y' = z
        y_s = z[k]
        # x' - y'**2 * np.cos(2*t + x) = 5*x**2 - 25*t**2
        x_s = y[k]**2 * np.cos(2*t[k] + x[k]) + 5*x[k]**2 - 25*t[k]**2
        # y'' = 4/(1 + y**2 * x**2) + 4 * np.sin(t * x')
        z_s = 4 / (1 + y[k]**2 * x[k]**2) + 4 * np.sin(t[k] * z[k])

        y[k+1] = y[k] + h * y_s
        x[k+1] = x[k] + h * x_s
        z[k+1] = z[k] + h * z_s
    return (z[N-1] - 0) ** 2


# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F512_square, F512_limits, d1=True, show=True, break_faster=True, count=300).run(save=f"Lab3/Images/{F512_square.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F512_square, F512_limits, d1=True, show=True, random=False, break_faster=True).run(save=f"Lab3/Images/{F512_square.__name__}_Bat.gif")


# --------------------------------------- #
def F513_square(X):
    N = 101
    xa, xb = 1, 3
    # h = (xb - xa) / (N - 1)
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    # x(1) = 2, y(1) = 1, y'(3) = -1, t = [1, 3]
    x[0] = 2
    y[0] = 1
    z[0] = X
    for k in range(N - 1):
        # y' = z
        y_s = z[k]

        # x' = 2*x**2 - 25*t**2 - np.sin(x * y * t)
        x_s = 2*x[k]**2 - 25*t[k]**2 - np.sin(x[k] * y[k] * t[k])

        # y'' = 1 - 4 * np.cos(x' * t)
        z_s = 1 - 4 * np.cos(x_s * t[k])

        y[k+1] = y[k] + h * y_s
        x[k+1] = x[k] + h * x_s
        z[k+1] = z[k] + h * z_s
    return (z[N-1] - -1) ** 2


# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F513_square, F513_limits, d1=True, show=True, break_faster=True, count=300, dots=1000).run(save=f"Lab3/Images/{F513_square.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F513_square, F513_limits, d1=True, show=True, random=False, break_faster=True, dots=1000, count=10).run(save=f"Lab3/Images/{F513_square.__name__}_Bat.gif")
# --------------------------------------- #
def F511_abs(X):
    N = 101
    xa, xb = 1, 3
    # h = (xb - xa) / (N - 1)
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    # x'(1) = 1, x(3) = 2, y(1) = 1, t = [1, 3]
    x[0] = X
    y[0] = 1
    z[0] = 1
    for k in range(N - 1):
        # x' = z
        x_s = z[k]
        # y' = y**4 + x**3 - 3 * np.sin(t * x')
        y_s = y[k]**4 + x[k]**3 - 3 * np.sin(t[k] * x_s)
        # x'' + y'**2 * np.cos(x') = x**2 * t**2    
        # x'' = x**2 * t**2 - y'**2 * np.cos(x')
        # z' = x**2 * t**2 - y'**2 * np.cos(z)
        z_s = x[k]**2 * t[k]**2 - y_s**2 * np.cos(z[k])

        y[k+1] = y[k] + h * y_s
        x[k+1] = x[k] + h * x_s
        z[k+1] = z[k] + h * z_s
    return abs(x[N-1] - 2)


# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F511_abs, F511_limits, d1=True, show=True, break_faster=True, count=300).run(save=f"Lab3/Images/{F511_abs.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F511_abs, F511_limits, d1=True, show=True, random=False, break_faster=True).run(save=f"Lab3/Images/{F511_abs.__name__}_Bat.gif")


# --------------------------------------- #
def F512_abs(X):
    N = 101
    xa, xb = 1, 3
    # h = (xb - xa) / (N - 1)
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    # x(1) = 2, y(1) = 2, y'(3) = 0, t = [1, 3]
    x[0] = 2
    y[0] = 2
    z[0] = X
    for k in range(N - 1):
        # y' = z
        y_s = z[k]
        # x' - y'**2 * np.cos(2*t + x) = 5*x**2 - 25*t**2
        x_s = y[k]**2 * np.cos(2*t[k] + x[k]) + 5*x[k]**2 - 25*t[k]**2
        # y'' = 4/(1 + y**2 * x**2) + 4 * np.sin(t * x')
        z_s = 4 / (1 + y[k]**2 * x[k]**2) + 4 * np.sin(t[k] * z[k])

        y[k+1] = y[k] + h * y_s
        x[k+1] = x[k] + h * x_s
        z[k+1] = z[k] + h * z_s
    return abs(z[N-1] - 0)


# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F512_abs, F512_limits, d1=True, show=True, break_faster=True, count=300).run(save=f"Lab3/Images/{F512_abs.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F512_abs, F512_limits, d1=True, show=True, random=False, break_faster=True).run(save=f"Lab3/Images/{F512_abs.__name__}_Bat.gif")


# --------------------------------------- #
def F513_abs(X):
    N = 101
    xa, xb = 1, 3
    # h = (xb - xa) / (N - 1)
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    # x(1) = 2, y(1) = 1, y'(3) = -1, t = [1, 3]
    x[0] = 2
    y[0] = 1
    z[0] = X
    for k in range(N - 1):
        # y' = z
        y_s = z[k]

        # x' = 2*x**2 - 25*t**2 - np.sin(x * y * t)
        x_s = 2*x[k]**2 - 25*t[k]**2 - np.sin(x[k] * y[k] * t[k])

        # y'' = 1 - 4 * np.cos(x' * t)
        z_s = 1 - 4 * np.cos(x_s * t[k])

        y[k+1] = y[k] + h * y_s
        x[k+1] = x[k] + h * x_s
        z[k+1] = z[k] + h * z_s
    return abs(z[N-1] - -1)


# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F513_abs, F513_limits, d1=True, show=True, break_faster=True, count=300, dots=1000).run(save=f"Lab3/Images/{F513_abs.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F513_abs, F513_limits, d1=True, show=True, random=False, break_faster=True, dots=1000, count=10).run(save=f"Lab3/Images/{F513_abs.__name__}_Bat.gif")

# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# 4. (2 додаткові бали) Модифікувати відповідні програмі для знаходження розв’язку
# будь-якої задачі з підрозділу 5.2. Показати динаміку збіжності обраного
# методу на обраній задачі, навести графічно відповідний розв’язок задач та
# числові параметри, які демонструють достатню збіжність методу на задачі. У
# якості цільової функції взяти:
#     a) f(α; β) = m|x(b; α; β) − A| + n|y(b; α; β) − B|;
#     b) f(α; β) = m(x(b; α; β) − A)**2 + n(y(b; α; β) − B)**2,
#       де b – значення аргументу на границі; A – потрібне значення функції
#       на границі t = b функції x(t) для пари (α; β); B – потрібне значення
#       функції на границі t = b функції y(t) для пари (α; β); m ∈ R+, n ∈ R+.
#     c) f(α; β) = max(m|x(b; α; β) − A|; n|y(b; α; β) − B|),
#     де b – значення аргументу на границі; A – потрібне значення функції
#     на границі t = b функції x(t) для пари (α; β); B – потрібне значення
#     функції на границі t = b функції y(t) для пари (α; β); m ∈ R+, n ∈ R+.


# --------------------------------------- #
@nb.jit()
def F521_plot():
    N = 1000
    xa, xb = 1, 3
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]
    al1 = np.arange(-8, 8, 0.1)
    al2 = np.arange(-8, 8, 0.1)

    x = np.zeros(N)
    y = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N)

    f = np.zeros((al1.shape[0], al2.shape[0]))

    # x(1) = 2, y(1) = -1, x(3) = 10, y(3) = 21, t = [1, 3]
    for l1 in range(len(al1)):
        for l2 in range(len(al2)):
            x[0] = 2  # al1[l1]
            y[0] = -1  # al2[l2]
            z1[0] = al1[l1]  # 2
            z2[0] = al2[l2]  # -1
            for k in range(N - 1):
                # x' = z1
                x_s = z1[k]
                # y' = z2
                y_s = z2[k]

                # x'' = np.cos(x * y) - (np.sin(y + t * x))/((x**2 + y')**2 + 1)
                z1_s = np.cos(x[k] * y[k]) - (np.sin(y[k] + t[k] * x[k]))/((x[k]**2 + y_s)**2 + 1)
                # y'' = 6*t - 1/t**4 + np.cos(5) - 4 + y'/t**2 - np.cos(3x-y') + x**2/t**4
                z2_s = 6*t[k] - 1/t[k]**4 + np.cos(5) - 4 + y_s/t[k]**2 - np.cos(3*x[k]-y_s) + x[k]**2/t[k]**4

                x[k+1] = x[k] + h * x_s
                y[k+1] = y[k] + h * y_s
                z1[k+1] = z1[k] + h * z1_s
                z2[k+1] = z2[k] + h * z2_s
            f[l1, l2] = ((x[N-1] - 10) ** 2 + (y[N-1] - 21) ** 2)
            # f[l1, l2] = (abs(x[N-1] - 10) ** 1 + abs(y[N-1] - 21) ** 1)
            # f[l1, l2] = max(abs(x[N-1] - 10) ** 1, abs(y[N-1] - 21) ** 1)

    return al1, al2, f


F521_limits = [[-8, 8], [-8, 8]]


# fig = plt.figure(figsize=plt.figaspect(2.))
# ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# plt.contourf(*F521_plot())
# plt.colorbar()
# plt.grid()
# plt.show()


# --------------------------------------- #
@nb.jit()
def F522_plot():
    N = 1000
    xa, xb = 1, 3
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]
    al1 = np.arange(-30, 30, 0.2)
    al2 = np.arange(-30, 30, 0.2)

    x = np.zeros(N)
    y = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N)

    f = np.zeros((al1.shape[0], al2.shape[0]))

    # x(1) = 2, y(1) = -1, x(3) = 5, y(3) = -1, t = [1, 3]
    for l1 in range(len(al1)):
        for l2 in range(len(al2)):
            x[0] = 2  # al1[l1]
            y[0] = -1  # al2[l2]
            z1[0] = al1[l1]  # 2
            z2[0] = al2[l2]  # -1
            for k in range(N - 1):
                # x' = z1
                x_s = z1[k]
                # y' = z2
                y_s = z2[k]

                # x'' = np.exp(-t**2) - 4 * np.exp(-abs(x * y_s)) * np.cos(np.sin(5*x_s**2*y + t**2)) - np.log(3*t**2 + (t * x + y_s)**2)
                z1_s = np.exp(-t[k]**2) - 4 * np.exp(-abs(x[k] * y_s)) * np.cos(np.sin(5*x_s**2*y[k] + t[k]**2)) - np.log(3*t[k]**2 + (t[k] * x[k] + y_s)**2)
                # y'' = np.exp(-t) * np.cos(2*t) + np.cos(abs((3*x - y_s)/(x_s**2 + 1))) - np.log(10 + np.exp(-abs(x**2 * y_s**3))) 
                z2_s = np.exp(-t[k]) * np.cos(2*t[k]) + np.cos(abs((3*x[k] - y_s)/(x_s**2 + 1))) - np.log(10 + np.exp(-abs(x[k]**2 * y_s**3)))

                x[k+1] = x[k] + h * x_s
                y[k+1] = y[k] + h * y_s
                z1[k+1] = z1[k] + h * z1_s
                z2[k+1] = z2[k] + h * z2_s
            f[l1, l2] = ((x[N-1] - 5) ** 2 + (y[N-1] + 1) ** 2)
            # f[l1, l2] = (abs(x[N-1] - 5) ** 1 + abs(y[N-1] + 1) ** 1)
            # f[l1, l2] = max(abs(x[N-1] - 5) ** 1, abs(y[N-1] + 1) ** 1)

    return al1, al2, f


F522_limits = [[-30, 30], [-30, 30]]


# fig = plt.figure(figsize=plt.figaspect(2.))
# ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# plt.contourf(*F522_plot())
# plt.colorbar()
# plt.grid()
# plt.show()


# --------------------------------------- #
@nb.jit()
def F521_square(X):
    N = 1000
    xa, xb = 1, 3
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]
    # al1 = np.arange(-8, 8, 0.2)
    # al2 = np.arange(-8, 8, 0.2)

    x = np.zeros(N)
    y = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N)

    # f = np.zeros((al1.shape[0], al2.shape[0]))

    # x(1) = 2, y(1) = -1, x(3) = 10, y(3) = 21, t = [1, 3]
    # x[0], y[0] = X
    # z1[0] = 2
    # z2[0] = -1
    x[0] = 2
    y[0] = -1
    z2[0], z1[0] = X
    for k in range(N - 1):
        # x' = z1
        x_s = z1[k]
        # y' = z2
        y_s = z2[k]

        # x'' = np.cos(x * y) - (np.sin(y + t * x))/((x**2 + y')**2 + 1)
        z1_s = np.cos(x[k] * y[k]) - (np.sin(y[k] + t[k] * x[k]))/((x[k]**2 + y_s)**2 + 1)
        # y'' = 6*t - 1/t**4 + np.cos(5) - 4 + y'/t**2 - np.cos(3x-y') + x**2/t**4
        z2_s = 6*t[k] - 1/t[k]**4 + np.cos(5) - 4 + y_s/t[k]**2 - np.cos(3*x[k]-y_s) + x[k]**2/t[k]**4

        y[k+1] = y[k] + h * y_s
        x[k+1] = x[k] + h * x_s
        z1[k+1] = z1[k] + h * z1_s
        z2[k+1] = z2[k] + h * z2_s
    return (x[N-1] - 10) ** 2 + (y[N-1] - 21) ** 2


# --------------------------------------- #
@nb.jit()
def F522_square(X):
    N = 1000
    xa, xb = 1, 3
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]

    x = np.zeros(N)
    y = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N)

    # x(1) = 2, y(1) = -1, x(3) = 5, y(3) = -1, t = [1, 3]
    x[0] = 2
    y[0] = -1
    z1[0], z2[0] = X
    for k in range(N - 1):
        # x' = z1
        x_s = z1[k]
        # y' = z2
        y_s = z2[k]

        # x'' = np.exp(-t**2) - 4 * np.exp(-abs(x * y_s)) * np.cos(np.sin(5*x_s**2*y + t**2)) - np.log(3*t**2 + (t * x + y_s)**2)
        z1_s = np.exp(-t[k]**2) - 4 * np.exp(-abs(x[k] * y_s)) * np.cos(np.sin(5*x_s**2*y[k] + t[k]**2)) - np.log(3*t[k]**2 + (t[k] * x[k] + y_s)**2)
        # y'' = np.exp(-t) * np.cos(2*t) + np.cos(abs((3*x - y_s)/(x_s**2 + 1))) - np.log(10 + np.exp(-abs(x**2 * y_s**3))) 
        z2_s = np.exp(-t[k]) * np.cos(2*t[k]) + np.cos(abs((3*x[k] - y_s)/(x_s**2 + 1))) - np.log(10 + np.exp(-abs(x[k]**2 * y_s**3)))

        x[k+1] = x[k] + h * x_s
        y[k+1] = y[k] + h * y_s
        z1[k+1] = z1[k] + h * z1_s
        z2[k+1] = z2[k] + h * z2_s
    return (x[N-1] - 10) ** 2 + (y[N-1] - 21) ** 2


# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F521_square, F521_limits, d2=True, show=True, break_faster=True, count=300, dots=100).run(save=f"Lab3/Images/{F521_square.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F521_square, F521_limits, d2=True, show=True, random=False, break_faster=True, dots=100, count=10).run(save=f"Lab3/Images/{F521_square.__name__}_Bat.gif")

# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F522_square, F522_limits, d2=True, show=True, break_faster=True, count=300, dots=100).run(save=f"Lab3/Images/{F522_square.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F522_square, F522_limits, d2=True, show=True, random=False, break_faster=True, dots=100, count=10).run(save=f"Lab3/Images/{F522_square.__name__}_Bat.gif")


# --------------------------------------- #
@nb.jit()
def F521_abs(X):
    N = 1000
    xa, xb = 1, 3
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]
    # al1 = np.arange(-8, 8, 0.2)
    # al2 = np.arange(-8, 8, 0.2)

    x = np.zeros(N)
    y = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N)

    # f = np.zeros((al1.shape[0], al2.shape[0]))

    # x(1) = 2, y(1) = -1, x(3) = 10, y(3) = 21, t = [1, 3]
    # x[0], y[0] = X
    # z1[0] = 2
    # z2[0] = -1
    x[0] = 2
    y[0] = -1
    z2[0], z1[0] = X
    for k in range(N - 1):
        # x' = z1
        x_s = z1[k]
        # y' = z2
        y_s = z2[k]

        # x'' = np.cos(x * y) - (np.sin(y + t * x))/((x**2 + y')**2 + 1)
        z1_s = np.cos(x[k] * y[k]) - (np.sin(y[k] + t[k] * x[k]))/((x[k]**2 + y_s)**2 + 1)
        # y'' = 6*t - 1/t**4 + np.cos(5) - 4 + y'/t**2 - np.cos(3x-y') + x**2/t**4
        z2_s = 6*t[k] - 1/t[k]**4 + np.cos(5) - 4 + y_s/t[k]**2 - np.cos(3*x[k]-y_s) + x[k]**2/t[k]**4

        y[k+1] = y[k] + h * y_s
        x[k+1] = x[k] + h * x_s
        z1[k+1] = z1[k] + h * z1_s
        z2[k+1] = z2[k] + h * z2_s
    return abs(x[N-1] - 10) + abs(y[N-1] - 21)


# --------------------------------------- #
@nb.jit()
def F522_abs(X):
    N = 1000
    xa, xb = 1, 3
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]

    x = np.zeros(N)
    y = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N)

    # x(1) = 2, y(1) = -1, x(3) = 5, y(3) = -1, t = [1, 3]
    x[0] = 2
    y[0] = -1
    z1[0], z2[0] = X
    for k in range(N - 1):
        # x' = z1
        x_s = z1[k]
        # y' = z2
        y_s = z2[k]

        # x'' = np.exp(-t**2) - 4 * np.exp(-abs(x * y_s)) * np.cos(np.sin(5*x_s**2*y + t**2)) - np.log(3*t**2 + (t * x + y_s)**2)
        z1_s = np.exp(-t[k]**2) - 4 * np.exp(-abs(x[k] * y_s)) * np.cos(np.sin(5*x_s**2*y[k] + t[k]**2)) - np.log(3*t[k]**2 + (t[k] * x[k] + y_s)**2)
        # y'' = np.exp(-t) * np.cos(2*t) + np.cos(abs((3*x - y_s)/(x_s**2 + 1))) - np.log(10 + np.exp(-abs(x**2 * y_s**3))) 
        z2_s = np.exp(-t[k]) * np.cos(2*t[k]) + np.cos(abs((3*x[k] - y_s)/(x_s**2 + 1))) - np.log(10 + np.exp(-abs(x[k]**2 * y_s**3)))

        x[k+1] = x[k] + h * x_s
        y[k+1] = y[k] + h * y_s
        z1[k+1] = z1[k] + h * z1_s
        z2[k+1] = z2[k] + h * z2_s
    return abs(x[N-1] - 10) ** 1 + abs(y[N-1] - 21) ** 1


# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F521_abs, F521_limits, d2=True, show=True, break_faster=True, count=300, dots=100).run(save=f"Lab3/Images/{F521_abs.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F521_abs, F521_limits, d2=True, show=True, random=False, break_faster=True, dots=100, count=10).run(save=f"Lab3/Images/{F521_abs.__name__}_Bat.gif")

# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F522_abs, F522_limits, d2=True, show=True, break_faster=True, count=300, dots=100).run(save=f"Lab3/Images/{F522_abs.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F522_abs, F522_limits, d2=True, show=True, random=False, break_faster=True, dots=100, count=10).run(save=f"Lab3/Images/{F522_abs.__name__}_Bat.gif")


# --------------------------------------- #
@nb.jit()
def F521_max(X):
    N = 1000
    xa, xb = 1, 3
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]
    # al1 = np.arange(-8, 8, 0.2)
    # al2 = np.arange(-8, 8, 0.2)

    x = np.zeros(N)
    y = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N)

    # f = np.zeros((al1.shape[0], al2.shape[0]))

    # x(1) = 2, y(1) = -1, x(3) = 10, y(3) = 21, t = [1, 3]
    # x[0], y[0] = X
    # z1[0] = 2
    # z2[0] = -1
    x[0] = 2
    y[0] = -1
    z2[0], z1[0] = X
    for k in range(N - 1):
        # x' = z1
        x_s = z1[k]
        # y' = z2
        y_s = z2[k]

        # x'' = np.cos(x * y) - (np.sin(y + t * x))/((x**2 + y')**2 + 1)
        z1_s = np.cos(x[k] * y[k]) - (np.sin(y[k] + t[k] * x[k]))/((x[k]**2 + y_s)**2 + 1)
        # y'' = 6*t - 1/t**4 + np.cos(5) - 4 + y'/t**2 - np.cos(3x-y') + x**2/t**4
        z2_s = 6*t[k] - 1/t[k]**4 + np.cos(5) - 4 + y_s/t[k]**2 - np.cos(3*x[k]-y_s) + x[k]**2/t[k]**4

        y[k+1] = y[k] + h * y_s
        x[k+1] = x[k] + h * x_s
        z1[k+1] = z1[k] + h * z1_s
        z2[k+1] = z2[k] + h * z2_s
    return max(abs(x[N-1] - 10), abs(y[N-1] - 21))


# --------------------------------------- #
@nb.jit()
def F522_max(X):
    N = 1000
    xa, xb = 1, 3
    t = np.linspace(xa, xb, N)
    h = t[1] - t[0]

    x = np.zeros(N)
    y = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N)

    # x(1) = 2, y(1) = -1, x(3) = 5, y(3) = -1, t = [1, 3]
    x[0] = 2
    y[0] = -1
    z1[0], z2[0] = X
    for k in range(N - 1):
        # x' = z1
        x_s = z1[k]
        # y' = z2
        y_s = z2[k]

        # x'' = np.exp(-t**2) - 4 * np.exp(-abs(x * y_s)) * np.cos(np.sin(5*x_s**2*y + t**2)) - np.log(3*t**2 + (t * x + y_s)**2)
        z1_s = np.exp(-t[k]**2) - 4 * np.exp(-abs(x[k] * y_s)) * np.cos(np.sin(5*x_s**2*y[k] + t[k]**2)) - np.log(3*t[k]**2 + (t[k] * x[k] + y_s)**2)
        # y'' = np.exp(-t) * np.cos(2*t) + np.cos(abs((3*x - y_s)/(x_s**2 + 1))) - np.log(10 + np.exp(-abs(x**2 * y_s**3))) 
        z2_s = np.exp(-t[k]) * np.cos(2*t[k]) + np.cos(abs((3*x[k] - y_s)/(x_s**2 + 1))) - np.log(10 + np.exp(-abs(x[k]**2 * y_s**3)))

        x[k+1] = x[k] + h * x_s
        y[k+1] = y[k] + h * y_s
        z1[k+1] = z1[k] + h * z1_s
        z2[k+1] = z2[k] + h * z2_s
    return max(abs(x[N-1] - 10) ** 1, abs(y[N-1] - 21) ** 1)


# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F521_max, F521_limits, d2=True, show=True, break_faster=True, count=300, dots=100).run(save=f"Lab3/Images/{F521_max.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F521_max, F521_limits, d2=True, show=True, random=False, break_faster=True, dots=100, count=10).run(save=f"Lab3/Images/{F521_max.__name__}_Bat.gif")

# Cuckoo(100, 1000, .5, 1, 1, [0, 1], F522_max, F522_limits, d2=True, show=True, break_faster=True, count=300, dots=100).run(save=f"Lab3/Images/{F521_max.__name__}_Cuckoo.gif")
# Bat(100, 1000, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], F522_max, F522_limits, d2=True, show=True, random=False, break_faster=True, dots=100, count=10).run(save=f"Lab3/Images/{F521_max.__name__}_Bat.gif")

# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
# 5. Оцінити графічно, наскільки ефективно працюють методи для різних цільових функцій.

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


# funcs = [(Rozenbrock1, Rozenbrock1_limits), (Rozenbrock2, Rozenbrock2_limits), (Siminonesku, Siminonesku_limits)]
# funcs = funcs + [(Reductor, Reductor_limits), (Trail, Trail_limits)]
# pop = 100
# iterat = 1000
# for func, limit in funcs:
#     Cuckoo(pop, iterat, .5, 1, 1, [0, 1], func, limit, d2=True, show=True, break_faster=False, count=300, dots=1000, plot=False).run(save=f"Lab3/Images/{func.__name__}_Cuckoo.gif")
#     Bat(pop, iterat, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], func, limit, d2=True, show=True, random=False, break_faster=False, plot=False, dots=1000, count=10).run(save=f"Lab3/Images/{func.__name__}_Bat.gif")  # savep=f"Lab3/Images/{func.__name__}.png"
#     plt.close()

#     Cuckoo(pop, iterat, .5, 1, 1, [0, 1], func, limit, d2=False, show=True, break_faster=False, count=300, dots=1000, plot=False, close=False).run()
#     Bat(pop, iterat, .5, 1.5, .5, .5, .5, [.1, .7], [0, 1], func, limit, d2=False, show=True, random=False, break_faster=False, dots=1000, count=10).run(savep=f"Lab3/Images/{func.__name__}.png")
