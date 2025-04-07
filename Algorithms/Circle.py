import numpy as np
import matplotlib.pyplot as plt


def Circle(N: int, plot: bool = False, random: bool = False):
    dfi = 2 * np.pi / N

    x = np.zeros(N)
    y = np.zeros(N)

    for i in range(N):
        x[i] = np.cos(i * dfi)
        y[i] = np.sin(i * dfi)

    if plot:
        plt.plot(x, y, "o")

    if random:
        way = np.array(range(N))
        np.random.shuffle(way)
        x = x[way]
        y = y[way]
    s = np.array([np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2) for i in range(N-1)])
    s = np.append(s, np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))
    if plot:
        plt.plot(x, y)
        plt.show()

    return np.append(np.append(x[:, np.newaxis], y[:, np.newaxis], axis=1), s[:, np.newaxis], axis=1)


if __name__ == "__main__":
    # b = 10
    # a = Circle(b, plot=False)
    # N = 100
    # # print(a)
    # for i in range(N):
    #     a = np.append(a, Circle(10, plot=False), axis=0)
    # for j in range(N+1):
    #     plt.plot(a[(j+0)*b:(j+1)*b, 0], a[(j+0)*b:(j+1)*b, 1])
    # plt.plot(a[:b, 0], a[:b, 1], "o")
    # # print(a)
    # plt.show()
    print(np.sum(Circle(10, plot=False)[:, 2]))
