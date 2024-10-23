import queue

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Node:
    def __init__(self, x, y, px, py, dis) -> None:
        self.x, self.y = x, y
        self.px, self.py = px, py
        self.dis = dis
        self.dll = []  # TODO: stored coordinates whose parent is this node


class ESDF:
    def __init__(self, gridmap: np.ndarray) -> None:
        """
        gridmap (np.ndarray): grid map, 0 for occupied, 1 for free
        """
        self.gridmap = gridmap

        self.rows, self.cols = self.gridmap.shape
        self.dist = np.full(self.gridmap.shape, np.inf)

        self.dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def cvDistanceTransform(self, ord=2) -> None:
        if ord == 1:
            self.dist = cv.distanceTransform(self.gridmap, cv.DIST_L1, 3)
        elif ord == 2:
            self.dist = cv.distanceTransform(self.gridmap, cv.DIST_L2, 3)
        else:
            raise NotImplementedError

    def bruteForce(self, ord=2) -> None:
        for i in tqdm(range(self.rows)):
            for j in range(self.cols):
                if self.gridmap[i][j] == 0:
                    self.dist[i][j] = 0
                    continue

                point_c = np.array([i, j])
                for ni in range(self.rows):
                    for nj in range(self.cols):
                        point_n = np.array([ni, nj])
                        if np.all(point_c == point_n):
                            continue

                        if self.gridmap[ni][nj] == 1:
                            # only find occupied grid
                            continue

                        self.dist[i][j] = min(self.dist[i][j], self.getDist(point_c, point_n, ord=ord))

    def forwardBFS(self) -> None:
        for i in tqdm(range(self.rows)):
            for j in range(self.cols):
                if self.gridmap[i][j] == 0:
                    self.dist[i][j] = 0
                    continue

                open = queue.Queue()
                closed = np.zeros_like(self.gridmap)

                open.put((i, j))

                step = 0
                found = False
                while not open.empty() and not found:
                    size = open.qsize()
                    for _ in range(size):
                        ci, cj = open.get()
                        if closed[ci][cj] == 1:
                            continue

                        closed[ci][cj] = 1
                        if self.gridmap[ci][cj] == 0:
                            # find occupied grid
                            self.dist[i][j] = step
                            found = True
                            break

                        # expand neighbours
                        for dir in self.dirs:
                            ni, nj = ci + dir[0], cj + dir[1]
                            if ni < 0 or ni >= self.rows or nj < 0 or nj >= self.cols or closed[ni][nj] == 1:
                                continue

                            open.put((ni, nj))

                    step += 1

    def backwardBFS(self) -> None:
        open = queue.Queue()
        closed = np.zeros_like(self.gridmap)

        for i in range(self.rows):
            for j in range(self.cols):
                if self.gridmap[i][j] == 0:
                    open.put((i, j))

        step = 0
        while not open.empty():
            size = open.qsize()
            for _ in range(size):
                ci, cj = open.get()
                if closed[ci][cj] == 1:
                    continue

                closed[ci][cj] = 1
                self.dist[ci][cj] = step

                # expand neighbours
                for dir in self.dirs:
                    ni, nj = ci + dir[0], cj + dir[1]
                    if ni < 0 or ni >= self.rows or nj < 0 or nj >= self.cols or closed[ni][nj] == 1:
                        continue

                    open.put((ni, nj))

            step += 1

    def sweepDP(self) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                if self.gridmap[i][j] == 0:
                    self.dist[i][j] = 0

        # from left-top to right-bottom
        for i in tqdm(range(self.rows)):
            for j in range(self.cols):
                if i - 1 >= 0:
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i - 1][j] + 1)

                if j - 1 >= 0:
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i][j - 1] + 1)

                # if i - 1 >= 0 and j - 1 >= 0:
                #     self.dist[i][j] = min(self.dist[i][j], self.dist[i - 1][j - 1] + np.sqrt(2))

        # from right-bottom to left-top
        for i in tqdm(range(self.rows - 1, -1, -1)):
            for j in range(self.cols - 1, -1, -1):
                if i + 1 < self.rows:
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i + 1][j] + 1)

                if j + 1 < self.cols:
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i][j + 1] + 1)

                # if i + 1 < self.rows and j + 1 < self.cols:
                #     self.dist[i][j] = min(self.dist[i][j], self.dist[i + 1][j + 1] + np.sqrt(2))

    def distanceTransform1D(self, pos, dim=0):
        if dim not in [0, 1]:
            raise NotImplementedError

        n = self.gridmap.shape[dim]
        f = self.dist[:, pos] if dim == 0 else self.dist[pos, :]

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
        edf = np.zeros(n)
        for q in range(n):
            while z[k + 1] < q:
                k += 1

            edf[q] = (q - v[k]) ** 2 + f[v[k]]

        if dim == 0:
            self.dist[:, pos] = edf
        else:
            self.dist[pos, :] = edf

    def distanceTransformSampleFunction(self) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                if self.gridmap[i][j] == 0:
                    self.dist[i][j] = 0

        for i in range(self.rows):
            self.distanceTransform1D(i, dim=1)

        for j in range(self.cols):
            self.distanceTransform1D(j, dim=0)

        self.dist = np.sqrt(self.dist)

    def backwardBFSL2(self) -> None:
        open = queue.PriorityQueue()
        closed = np.zeros_like(self.gridmap)
        node_map = [[Node(i, j, np.inf, np.inf, np.inf) for j in range(self.cols)] for i in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                if self.gridmap[i][j] == 0:
                    node_map[i][j].dis, node_map[i][j].px, node_map[i][j].py = 0, i, j
                    open.put((0, i, j))

        while not open.empty():
            dis, ci, cj = open.get()
            # print(f"ci = {ci}, cj = {cj}, dis = {dis}, closed = {closed[ci][cj]}")
            if closed[ci][cj] == 1:
                continue

            closed[ci][cj] = 1
            self.dist[ci][cj] = dis

            pi, pj = node_map[ci][cj].px, node_map[ci][cj].py

            # expand neighbours
            for dir in self.dirs:
                ni, nj = ci + dir[0], cj + dir[1]
                if ni < 0 or ni >= self.rows or nj < 0 or nj >= self.cols or closed[ni][nj] == 1:
                    continue

                dis = self.getDist(np.array([ni, nj]), np.array([pi, pj]), ord=2)
                if dis < node_map[ni][nj].dis:
                    node_map[ni][nj].dis, node_map[ni][nj].px, node_map[ni][nj].py = dis, pi, pj
                    open.put((dis, ni, nj))

    @staticmethod
    def getDist(point_1: np.ndarray, point_2: np.ndarray, ord: int = 2) -> float:
        return np.linalg.norm(point_1 - point_2, ord=ord)

    def updateESDF(self) -> None:
        # dist = self.dist

        # self.cvDistanceTransform(ord=1)
        # self.cvDistanceTransform(ord=2)

        # cv_dist = self.dist
        # self.dist = dist

        # self.bruteForce(ord=1)
        # self.bruteForce(ord=2)
        # self.forwardBFS()
        # self.backwardBFS()
        # self.sweepDP()
        # self.distanceTransformSampleFunction()
        self.backwardBFSL2()

        # al_dist = self.dist
        # self.dist = dist

        # print(np.sum(np.linalg.norm(cv_dist - al_dist)))

    def show(self) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(self.gridmap)
        axs[1].imshow(self.dist)

        plt.tight_layout()
        plt.show()


def main():
    # gridmap_file = "gridmap.bmp"
    gridmap_file = "gridmap_64x48.bmp"

    gridmap = cv.imread(gridmap_file, cv.IMREAD_GRAYSCALE)
    _, gridmap = cv.threshold(gridmap, 127, 1, cv.THRESH_BINARY)
    # cv.imshow("map", gridmap)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    esdf = ESDF(gridmap)
    esdf.updateESDF()
    esdf.show()


if __name__ == "__main__":
    main()
