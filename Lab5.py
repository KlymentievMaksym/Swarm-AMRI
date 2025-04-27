import numpy as np
from os import listdir

from Algorithms.Bag.Anneal import Anneal
from Algorithms.Bag.Genetic import Genetic

path = "."
if "Lab5" in listdir(path):
    path += "/Lab5"
    if "Data" in listdir(path):
        path += "/Data"
        datas_path = [path + "/" + data for data in listdir(path)]
    else:
        raise Exception("Data directory not found")
else:
    raise Exception("Lab5 directory not found")

for data_path in datas_path:
    with open(data_path, 'r') as fr:
        text = fr.read()

    data = text.split("\n")
    max_weight, data = int(data[0]), data[1:-1]
    for i in range(len(data)):
        data[i] = data[i].split(" ")
        data[i] = list(map(int, data[i]))
    data = np.array(data)
    iterations = 100
    ann = Anneal(iterations, 1e-3, 100, 0.99, max_weight, data, plot_convergence=True, plot_bar=True)
    gen = Genetic(iterations, 20, 100, 0.9, max_weight, data, plot_convergence=True, plot_bar=True)
    print(f"[Path] {data_path}")
    print(f"[Weight] Max weight: {max_weight}")
    print(f"[Ann] Best value: {ann[0]}, used weight: {np.sum(ann[1] * data[:, 0])}\n[Gen] Best value: {gen[0]}, used weight: {np.sum(gen[1] * data[:, 0])}\n")
