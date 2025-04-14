import numpy as np
from os import listdir

from Algorithms.TSP.GeneticRoute import GeneticRoute
# from Algorithms.AntRoute import AntRoute
# from Algorithms.Depreceated.AntTSP import AntTSP
from Algorithms.TSP.AntTSP import AntTSP

# print(listdir("./Lab4/Data"))
for path in listdir("./Lab4/Data"):
    with open(f"./Lab4/Data/{path}", 'r') as fr:
        text = fr.read()

    info_data = text.split("NODE_COORD_SECTION")

    info = info_data[0]
    info = info.split("\n")
    info = info[:-1]
    info = {part.split(" ")[0]: part.split(" ")[2] for part in info}

    data = info_data[1]
    data = data.split("\n")
    data = data[1:-2]
    data = np.array([list(map(int, part.split(" "))) for part in data])

    # GeneticRoute(100, 100, 500, 0.5, data[:, 1:]).run(show_plot_animation=False)  # , save_plot=f"./Lab4/Images/an.{path}.png", save_convergence=f"./Lab4/Images/conv.{path}.png")  #, save_plot="./Lab4/Images/an.3.png", save_convergence="./Lab4/Images/conv.3.png"
    AntTSP(data[:, 1:], 10, 100, 5, 5, 0.5, 100)
    # AntRoute(100, 100, 0.5, 0.5, 0.5, 280, data[:, 1:]).run(show_plot=True, show_plot_animation=False, show_convergence=True)  # , save_plot=f"./Lab4/Images/an.{path}.png", save_convergence=f"./Lab4/Images/conv.{path}.png")  #, save_plot="./Lab4/Images/an.3.png", save_convergence="./Lab4/Images/conv.3.png"

# from Algorithms.AntRoute import AntRoute
# from Algorithms.PSO import PSO
# def Ant(X):
#     x1, x2, x3, x4 = X
#     return AntRoute(20, 100, x1, x2, x3, x4).run(plot=False)[0]

# Ant_limits = [(0, 1), (0, 1), (0, 1), (0.1, 100)]

# pso = PSO(10, 10, [0, 4], [-.1, .1], Ant, Ant_limits).run()
# print(*pso)
