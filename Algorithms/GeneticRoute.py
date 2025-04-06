import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 10
P1 = np.arange(0, N, 1)
np.random.shuffle(P1)
# print(P1)
P2 = np.arange(0, N, 1)
np.random.shuffle(P2)

# Crossover
cross = np.random.randint(1, N)
C1 = np.concatenate((P1[:cross], P2[cross:], P2[:cross], P1[cross:]))
# print(C1)
# print()
C1 = C1[sorted(np.unique(C1, return_index=True)[1])]
# print(P1)
# print(P2)
# print(cross)
# print(C1)

# Mutation
cros1 = np.random.randint(N)
cros2 = np.random.randint(N)

while cros1 == cros2:
    cros2 = np.random.randint(N)

if cros1 > cros2:
    cros1, cros2 = cros2, cros1

# print(C1)
if np.random.rand() < 0.5:
    C1[cros1], C1[cros2] = C1[cros2], C1[cros1]
else:
    C1[cros1:cros2] = C1[cros1:cros2][::-1]
# print(C1)

# Population
pop_size = 20
parts = np.zeros((pop_size, N), dtype=int)
for i in range(pop_size):
    parts[i] = np.arange(0, N, 1)
    np.random.shuffle(parts[i])
print(parts)
