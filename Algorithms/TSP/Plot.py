import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation


class Plot:
    def define_kwargs(self, **kwargs):
        self.plot_do = kwargs.get("plot", True)

        self.show_every = kwargs.get("every", 1)
        self.interval = kwargs.get("interval", 30)
        self.fps = kwargs.get("fps", 30)

        self.plot_convergence = kwargs.get("plot_convergence", True)
        self.show_convergence = kwargs.get("show_convergence", True)
        self.show_convergence_animation = kwargs.get("show_convergence_animation", False)
        self.close_convergence = kwargs.get("close_convergence", not self.show_convergence)
        self.save_convergence = kwargs.get("save_convergence", False)

        self.plot_plot = kwargs.get("plot_plot", True)
        self.show_plot = kwargs.get("show_plot", True)
        self.show_plot_animation = kwargs.get("show_plot_animation", False)
        self.close_plot = kwargs.get("close_plot", not self.show_plot)
        self.save_plot = kwargs.get("save_plot", False)

    def plotTSP(self, history_best_f, history_best_routes, cities, **kwargs):
        self.define_kwargs(**kwargs)
        if self.plot_convergence:
            if not self.show_convergence_animation:
                plt.title(f"best fit {history_best_f[-1]:.4f}")
                plt.plot(np.arange(0, len(history_best_f), 1, dtype=int) * self.show_every, history_best_f)
                plt.grid()
                # plt.legend()
                plt.xlabel("Iterations")
                plt.ylabel("Fitness")

                if self.save_convergence:
                    try:
                        plt.savefig(self.save_convergence)
                    except Exception as e:
                        print(e)
            else:
                def update(frame):
                    plt.cla()
                    plt.title(f"iteration {(frame + 1) * self.show_every}, best fit {history_best_f[frame]:.4f}")
                    plt.plot(np.arange(0, len(history_best_f[:frame]), 1, dtype=int) * self.show_every, history_best_f[:frame])
                    plt.grid()
                    # plt.legend()
                    plt.xlabel("Iterations")
                    plt.ylabel("Fitness")
                ani = animation.FuncAnimation(plt.gcf(), update, frames=len(history_best_f), interval=self.interval)

                if self.save_convergence:
                    try:
                        ani.save(self.save_convergence, fps=self.fps)
                    except Exception as e:
                        print(e)

            if self.show_convergence:
                plt.show()
            elif self.close_convergence:
                plt.close()

        if self.plot_plot:
            if not self.show_plot_animation:
                # plt.cla()
                plt.title(f"best fit {history_best_f[-1]:.4f}")
                plt.grid()
                x = cities[history_best_routes[-1], 0]
                x = np.append(x, x[0])
                y = cities[history_best_routes[-1], 1]
                y = np.append(y, y[0])
                plt.plot(x, y, 'm')
                plt.plot(x, y, 'r.', markersize=15)
                plt.xlabel("x")
                plt.ylabel("y")

                if self.save_plot:
                    try:
                        plt.savefig(self.save_plot)
                    except Exception as e:
                        print(e)
            else:
                def update(frame):
                    plt.cla()
                    plt.title(f"iteration {(frame + 1) * self.show_every}, best fit {history_best_f[frame]:.4f}")
                    x = cities[history_best_routes[frame], 0]
                    x = np.append(x, x[0])
                    y = cities[history_best_routes[frame], 1]
                    y = np.append(y, y[0])
                    plt.plot(x, y, 'm')
                    plt.plot(x, y, 'r.', markersize=15)
                    plt.xlabel("x")
                    plt.ylabel("y")
                    # plt.plot(history_best_f[frame])
                ani = animation.FuncAnimation(plt.gcf(), update, frames=len(history_best_f), interval=self.interval)

                if self.save_plot:
                    try:
                        ani.save(self.save_plot, fps=self.fps)
                    except Exception as e:
                        print(e)

            if self.show_plot:
                plt.show()
            elif self.close_plot:
                plt.close()
