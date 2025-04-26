import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def Plot(history_pop, history_fitness, **kwargs):
    
    def update(frame):
        plt.cla()
        plt.title(f"iteration {frame}, best fit {history_fitness[frame][0]:.4f}")
        plt.bar(range(len(history_pop[frame])), history_fitness[frame])
        plt.xlabel("Part index")
        plt.ylabel("Value")
    ani = animation.FuncAnimation(plt.gcf(), update, frames=len(history_pop), interval=100)
    plt.show()
    # for i in range(len(history_pop[-1])):
    #     plt.bar(i, history_fitness[-1])
    # plt.xlabel("Part index")
    # plt.ylabel("Value")
    # plt.show()
