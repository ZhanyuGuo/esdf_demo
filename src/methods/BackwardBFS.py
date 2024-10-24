import queue

import numpy as np

from ESDF import ESDF


class BackwardBFS(ESDF):
    def __init__(self, grid: np.ndarray, has_vis=True, vis_interval=0.01, acc_rate=50, frame_rate=24, output_path="") -> None:
        super().__init__(grid=grid, has_vis=has_vis, vis_interval=vis_interval, acc_rate=acc_rate, frame_rate=frame_rate, output_path=output_path)

    def updateESDF(self) -> None:
        open = queue.Queue()
        closed = np.zeros_like(self.grid)

        for x in range(self.rows):
            for y in range(self.cols):
                if self.grid[x][y] == 0:
                    open.put((x, y))

        step = 0
        while not open.empty():
            size = open.qsize()
            for _ in range(size):
                cx, cy = open.get()
                if closed[cx][cy] == 1:
                    continue

                closed[cx][cy] = 1
                self.dist[cx][cy] = step
                self.updateDistFig()

                # expand neighbours
                for dir in self.dirs:
                    nx, ny = cx + dir[0], cy + dir[1]
                    if not self.checkPos(nx, ny) or closed[nx][ny] == 1:
                        continue

                    open.put((nx, ny))

            step += 1
