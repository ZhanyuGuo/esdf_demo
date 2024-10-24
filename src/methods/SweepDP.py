import numpy as np

from ESDF import ESDF


class SweepDP(ESDF):
    def __init__(self, grid: np.ndarray, has_vis=True, vis_interval=0.01, acc_rate=50, frame_rate=24, output_path="") -> None:
        super().__init__(grid=grid, has_vis=has_vis, vis_interval=vis_interval, acc_rate=acc_rate, frame_rate=frame_rate, output_path=output_path)

    def updateESDF(self) -> None:
        for x in range(self.rows):
            for y in range(self.cols):
                if self.grid[x][y] == 0:
                    self.dist[x][y] = 0
                    self.updateDistFig()

        # from left-top to right-bottom
        for x in range(self.rows):
            for y in range(self.cols):
                if x - 1 >= 0:
                    self.dist[x][y] = min(self.dist[x][y], self.dist[x - 1][y] + 1)

                if y - 1 >= 0:
                    self.dist[x][y] = min(self.dist[x][y], self.dist[x][y - 1] + 1)

                self.updateDistFig()

                # if i - 1 >= 0 and j - 1 >= 0:
                #     self.dist[i][j] = min(self.dist[i][j], self.dist[i - 1][j - 1] + np.sqrt(2))

        # from right-bottom to left-top
        for x in range(self.rows - 1, -1, -1):
            for y in range(self.cols - 1, -1, -1):
                if x + 1 < self.rows:
                    self.dist[x][y] = min(self.dist[x][y], self.dist[x + 1][y] + 1)

                if y + 1 < self.cols:
                    self.dist[x][y] = min(self.dist[x][y], self.dist[x][y + 1] + 1)

                self.updateDistFig()

                # if i + 1 < self.rows and j + 1 < self.cols:
                #     self.dist[i][j] = min(self.dist[i][j], self.dist[i + 1][j + 1] + np.sqrt(2))
