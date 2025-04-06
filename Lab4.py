import numpy as np

import matplotlib.pyplot as plt

with open("./Lab4/Data/xqf131.tsp", 'r') as fr:
    text = fr.read()


info_data = text.split("NODE_COORD_SECTION")

# print(text)

info = info_data[0]
info = info.split("\n")
info = info[:-1]
info = {part.split(" ")[0]: part.split(" ")[2] for part in info}

data = info_data[1]
data = data.split("\n")
data = data[1:-2]
data = [part.split(" ") for part in data]
data = [list(map(int, (i, j, value))) for value, i, j in data]

# print(data)
ofset = 1
i_max = max(data, key=lambda x: x[0])[0] + 1 + 2 * ofset
j_max = max(data, key=lambda x: x[1])[1] + 1 + 2 * ofset

dim = int(info.get("DIMENSION"))
matrix = np.zeros((i_max, j_max))

# for value, j, i in data:
for i, j, value in data:
    matrix[i+ofset, j+ofset] = value

# print(info)
# print(data)
# print(matrix)

# plt.imshow(matrix>0, cmap="binary")
# plt.show()

N = 4
dfi = 2 * np.pi / N

x = np.zeros(N)
y = np.zeros(N)

for i in range(N):
    x[i] = np.cos(i * dfi)
    y[i] = np.sin(i * dfi)

plt.plot(x, y, "o")
plt.show()
