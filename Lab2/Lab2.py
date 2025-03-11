from Algorithms.Bee import BEE
from Algorithms.Firefly import FF
from Algorithms.PSO import PSO


def F(X):
    x, y, = X
    f1 = x**2+y**2 < 2
    if f1:
        return (1 - x)**2 + 100*(y - x**2)**2
    else:
        return float('inf')

def Opt(X):
    # print(X.astype(int))
    return FF(*X.astype(int), [0, 1], F, [[-1.5, 1.5], [-1.5, 1.5]], progress=False).run()[0]
    # return BEE(*X.astype(int), [0, 1], F, [[-1.5, 1.5], [-1.5, 1.5]]).run()[0]

# bee = BEE(160, 44, 33, 10, 6, 2, [0, 1], F, [[-1.5, 1.5], [-1.5, 1.5]], d2=True, show=True, plot=True).run()
# ff = FF(50, 50, [0, 1], F, [[-1.5, 1.5], [-1.5, 1.5]], d2=True, show=True).run()
# pso = PSO(40, 70, [0, 4], [-.1, .1], Opt, [[50, 200], [10, 50], [10, 40], [5, 20], [2, 10], [1, 5]], d2=False, show=True, integer=[range(6)]).run()
pso = PSO(10, 10, [0, 4], [-.1, .1], Opt, [[50, 100], [10, 50]], d2=False, show=True, integer=[range(2)]).run()
print(*pso)
