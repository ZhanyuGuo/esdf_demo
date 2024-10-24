import queue

import numpy as np

from ESDF import ESDF


class DistanceTransform(ESDF):
    def __init__(self, grid: np.ndarray, has_vis=True, vis_interval=0.01, acc_rate=50, frame_rate=24, output_path="") -> None:
        super().__init__(grid=grid, has_vis=has_vis, vis_interval=vis_interval, acc_rate=acc_rate, frame_rate=frame_rate, output_path=output_path)

        self.square_dist = np.full(self.grid.shape, np.inf)


    def distanceTransform1D(self, pos, dim=0):
        if dim not in [0, 1]:
            raise NotImplementedError

        n = self.grid.shape[dim]
        f = self.square_dist[:, pos] if dim == 0 else self.square_dist[pos, :]

        k = 0
        v, z = np.zeros(n, dtype=np.int32), np.zeros(n + 1, dtype=np.float64)
        z[0], z[1] = -np.inf, np.inf

        for q in range(1, n):
            s = 0 if f[q] == np.inf and f[v[k]] == np.inf else ((f[q] + q**2) - (f[v[k]] + v[k] ** 2)) / (2 * (q - v[k]))
            while s <= z[k]:
                k -= 1
                s = 0 if f[q] == np.inf and f[v[k]] == np.inf else ((f[q] + q**2) - (f[v[k]] + v[k] ** 2)) / (2 * (q - v[k]))
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = np.inf

        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1

            if dim == 0:
                self.square_dist[q, pos] = (q - v[k]) ** 2 + f[v[k]]
                self.dist[q, pos] = np.sqrt(self.square_dist[q, pos])
            else:
                self.square_dist[pos, q] = (q - v[k]) ** 2 + f[v[k]]
                self.dist[pos, q] = np.sqrt(self.square_dist[pos, q])

            self.updateDistFig()

    def updateESDF(self) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 0:
                    self.dist[i][j] = 0
                    self.square_dist[i][j] = 0
                    self.updateDistFig()

        for i in range(self.rows):
            self.distanceTransform1D(i, dim=1)

        for j in range(self.cols):
            self.distanceTransform1D(j, dim=0)

        # self.dist = np.sqrt(self.dist)
