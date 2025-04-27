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

        self.plot_bar = kwargs.get("plot_bar", True)
        self.show_bar = kwargs.get("show_bar", True)
        self.show_bar_animation = kwargs.get("show_bar_animation", False)
        self.close_bar = kwargs.get("close_bar", not self.show_bar)
        self.save_bar = kwargs.get("save_bar", False)

    def Plot(self, history_pop, history_fitness, **kwargs):
        self.define_kwargs(**kwargs)
        convergence_do = self.plot_convergence and (self.show_convergence or not self.close_convergence)
        bar_do = self.plot_bar and (self.show_bar or not self.close_bar)
        if convergence_do and bar_do:
            self.PlotBoth(history_pop, history_fitness)
        else:
            if convergence_do:
                self.PlotConvergence(history_fitness)

            if bar_do:
                self.PlotBar(history_pop, history_fitness)

    def PlotBoth(self, history_pop, history_fitness):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        history_fitnes = history_fitness[:, 0]
        if not self.show_convergence_animation:
            ax1.set_title(f"best fit {history_fitness[-1][0]:.4f}")
            ax1.plot(np.arange(len(history_fitnes), step=1) * self.show_every, history_fitnes)
            ax1.grid()
            ax1.set_xlabel("Iterations")
            ax1.set_ylabel("Fitness")
        if not self.show_bar_animation:
            ax2.set_title(f"best fit {history_fitness[-1][0]:.4f}")
            ax2.bar(range(len(history_pop[-1])), history_fitness[-1])
            ax2.set_xlabel("Part index")
            ax2.set_ylabel("Value")

        padding = 5
        if self.show_convergence_animation and self.show_bar_animation:
            def update(frame):
                ax1.cla()
                ax1.set_title(f"iteration {frame * self.show_every}, best fit {history_fitness[frame][0]:.4f}")
                ax1.plot(np.arange(len(history_fitnes[:frame+1]), step=1) * self.show_every, history_fitnes[:frame+1])
                ax1.grid()
                ax1.set_xlabel("Iterations")
                ax1.set_ylabel("Fitness")
                # padding = 1
                ax1.set_xlim(0 - padding, len(history_fitnes) * self.show_every + padding)
                ax1.set_ylim(min(history_fitnes) - padding, max(history_fitnes) + padding)
                ax2.cla()
                ax2.set_title(f"iteration {frame * self.show_every}, best fit {history_fitness[frame][0]:.4f}")
                ax2.bar(range(len(history_pop[frame])), history_fitness[frame])
                ax2.set_xlabel("Part index")
                ax2.set_ylabel("Value")
                # padding = 5
                ax2.set_xlim(0 - padding, len(history_pop[frame]) + padding)
                ax2.set_ylim(0 - padding, history_fitness[-1][0] + padding)
        elif self.show_convergence_animation:
            def update(frame):
                ax1.cla()
                ax1.set_title(f"iteration {frame * self.show_every}, best fit {history_fitness[frame][0]:.4f}")
                ax1.plot(np.arange(len(history_fitnes[:frame+1]), step=1) * self.show_every, history_fitnes[:frame+1])
                ax1.grid()
                ax1.set_xlabel("Iterations")
                ax1.set_ylabel("Fitness")
                # padding = 1
                ax1.set_xlim(0 - padding, len(history_fitnes) * self.show_every + padding)
                ax1.set_ylim(min(history_fitnes) - padding, max(history_fitnes) + padding)
        elif self.show_bar_animation:
            def update(frame):
                ax2.cla()
                ax2.set_title(f"iteration {frame * self.show_every}, best fit {history_fitness[frame][0]:.4f}")
                ax2.bar(range(len(history_pop[frame])), history_fitness[frame])
                ax2.set_xlabel("Part index")
                ax2.set_ylabel("Value")
                # padding = 5
                ax2.set_xlim(0 - padding, len(history_pop[frame]) + padding)
                ax2.set_ylim(0 - padding, history_fitness[-1][0] + padding)
        if self.show_convergence_animation or self.show_bar_animation:
            ani = animation.FuncAnimation(fig, update, frames=len(history_pop), interval=self.interval)
            try:
                if self.save_convergence:
                    ani.save(self.save_convergence, fps=self.fps)
            except Exception as e:
                try:
                    if self.save_bar:
                        ani.save(self.save_bar, fps=self.fps)
                except Exception as e2:
                    print(e, e2)
        else:
            try:
                if self.save_convergence:
                    plt.savefig(self.save_convergence)
            except Exception as e:
                try:
                    if self.save_bar:
                        plt.savefig(self.save_bar)
                except Exception as e2:
                    print(e, e2)
        plt.show()

    def PlotConvergence(self, history_fitness):
        history_fitnes = history_fitness[:, 0]
        if not self.show_convergence_animation:
            plt.title(f"best fit {history_fitnes[-1]:.4f}")
            plt.plot(np.arange(len(history_fitnes), step=1) * self.show_every, history_fitnes)
            plt.grid()
            plt.xlabel("Iterations")
            plt.ylabel("Fitness")
        else:
            def update(frame):
                plt.cla()
                plt.title(f"iteration {frame * self.show_every}, best fit {history_fitnes[frame]:.4f}")
                plt.plot(np.arange(len(history_fitnes[:frame+1]), step=1) * self.show_every, history_fitnes[:frame+1])
                plt.grid()
                plt.xlabel("Iterations")
                plt.ylabel("Fitness")
                padding = 1
                plt.xlim(0 - padding, len(history_fitnes) * self.show_every + padding)
                plt.ylim(min(history_fitnes) - padding, max(history_fitnes) + padding)
            ani = animation.FuncAnimation(plt.gcf(), update, frames=len(history_fitnes), interval=self.interval)
        plt.show()

    def PlotBar(self, history_pop, history_fitness):
        if not self.show_bar_animation:
            plt.title(f"best fit {history_fitness[-1][0]:.4f}")
            plt.bar(range(len(history_pop[-1])), history_fitness[-1])
            plt.xlabel("Part index")
            plt.ylabel("Value")

            if self.save_bar:
                try:
                    plt.savefig(self.save_bar)
                except Exception as e:
                    print(e)
        else:
            def update(frame):
                plt.cla()
                plt.title(f"iteration {frame * self.show_every}, best fit {history_fitness[frame][0]:.4f}")
                plt.bar(range(len(history_pop[frame])), history_fitness[frame])
                plt.xlabel("Part index")
                plt.ylabel("Value")
                padding = 5
                plt.xlim(0 - padding, len(history_pop[frame]) + padding)
                plt.ylim(0 - padding, history_fitness[-1][0] + padding)
            ani = animation.FuncAnimation(plt.gcf(), update, frames=len(history_pop), interval=self.interval)
            if self.save_bar:
                try:
                    ani.save(self.save_bar, fps=self.fps)
                except Exception as e:
                    print(e)
        plt.show()
