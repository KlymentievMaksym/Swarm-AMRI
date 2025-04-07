import numpy as np

from Algorithms.GeneticRoute import GeneticRoute

with open("./Lab4/Data/xqf131.tsp", 'r') as fr:
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

GeneticRoute(1000, 5000, 5000, 0.5, data[:, 1:]).run(show_plot=False, show_plot_animation=True, show_convergence=False, save_plot="./Lab4/Images/an.4.gif", save_convergence="./Lab4/Images/conv.4.png")  #, save_plot="./Lab4/Images/an.3.png", save_convergence="./Lab4/Images/conv.3.png"
