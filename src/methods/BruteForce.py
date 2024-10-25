import numpy as np

from ESDF import ESDF


class BruteForce(ESDF):
    def __init__(self, grid: np.ndarray, has_vis=True, vis_interval=0.01, frame_interval=100, frame_rate=30, output_path="") -> None:
        super().__init__(grid=grid, has_vis=has_vis, vis_interval=vis_interval, frame_interval=frame_interval, frame_rate=frame_rate, output_path=output_path)

    def updateESDF(self, ord=2) -> None:
        for x in range(self.rows):
            for y in range(self.cols):
                if self.grid[x][y] == 0:
                    self.dist[x][y] = 0
                    continue

                p = np.array([x, y])
                for nx in range(self.rows):
                    for ny in range(self.cols):
                        if self.grid[nx][ny] == 1:
                            continue

                        o = np.array([nx, ny])
                        self.dist[x][y] = min(self.dist[x][y], self.getDist(p, o, ord=ord))

                self.updateDistFig()
