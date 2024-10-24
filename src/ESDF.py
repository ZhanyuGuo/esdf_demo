import os

import imageio
import matplotlib.pyplot as plt
import numpy as np


class ESDF:
    def __init__(self, grid: np.ndarray, has_vis=True, vis_interval=0.01, acc_rate=50, frame_rate=60, output_path="") -> None:
        self.grid = grid
        self.dist = np.full(self.grid.shape, np.inf)

        self.rows, self.cols = self.grid.shape

        self.dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

        self.has_vis = has_vis
        self.vis_interval = vis_interval
        self.acc_rate = acc_rate
        self.frame_rate = frame_rate
        self.output_file = os.path.join(output_path, self.__class__.__name__ + ".gif")

        if self.has_vis:
            self.cnt = 0
            self.frames = []
            plt.ion()
            self.fig, self.axs = plt.subplots(2, 1, figsize=(5, 8.5))
            self.axs[0].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            self.axs[1].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            self.axs[0].imshow(self.grid, cmap="gray")
            self.axs[1].imshow(self.dist)
            plt.pause(self.vis_interval)

    def updateDistFig(self, forced=False) -> None:
        if not self.has_vis:
            return

        self.cnt += 1
        if not forced and self.cnt % self.acc_rate:
            return

        self.axs[1].cla()
        self.axs[1].imshow(self.dist)
        plt.pause(self.vis_interval)
        self.frames.append(np.array(self.fig.canvas.renderer.buffer_rgba()))

    def show(self) -> None:
        if not self.has_vis:
            return

        self.updateDistFig(forced=True)

        plt.ioff()
        plt.show()

        imageio.mimsave(self.output_file, self.frames, fps=60, loop=0)

    @staticmethod
    def getDist(point_1: np.ndarray, point_2: np.ndarray, ord: int = 2) -> float:
        return np.linalg.norm(point_1 - point_2, ord=ord)

    def checkPos(self, x, y) -> bool:
        return x >= 0 and x < self.rows and y >= 0 and y < self.cols
