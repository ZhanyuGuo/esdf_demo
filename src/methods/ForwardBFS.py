import queue

import numpy as np

from ESDF import ESDF


class ForwardBFS(ESDF):
    def __init__(self, grid: np.ndarray, has_vis=True, vis_interval=0.01, frame_interval=100, frame_rate=30, output_path="") -> None:
        super().__init__(grid=grid, has_vis=has_vis, vis_interval=vis_interval, frame_interval=frame_interval, frame_rate=frame_rate, output_path=output_path)

    def updateESDF(self) -> None:
        for x in range(self.rows):
            for y in range(self.cols):
                if self.grid[x][y] == 0:
                    self.dist[x][y] = 0
                    self.updateDistFig()
                    continue

                open = queue.Queue()
                closed = np.zeros_like(self.grid)

                open.put((x, y))

                step, found = 0, False
                while not open.empty() and not found:
                    size = open.qsize()
                    for _ in range(size):
                        cx, cy = open.get()
                        if closed[cx][cy] == 1:
                            continue

                        closed[cx][cy] = 1
                        if self.grid[cx][cy] == 0:
                            self.dist[x][y], found = step, True
                            self.updateDistFig()
                            break

                        # expand neighbours
                        for dir in self.dirs:
                            nx, ny = cx + dir[0], cy + dir[1]
                            if not self.checkPos(nx, ny) or closed[nx][ny] == 1:
                                continue

                            open.put((nx, ny))

                    step += 1
