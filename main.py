import queue

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class ESDF:
    def __init__(self, map: np.ndarray) -> None:
        self.map = map

        self.rows, self.cols = self.map.shape
        self.dist = np.full(self.map.shape, np.inf)

        self.dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def cvDistanceTransform(self, ord=2) -> None:
        if ord == 1:
            self.dist = cv.distanceTransform(self.map, cv.DIST_L1, 3)
        elif ord == 2:
            self.dist = cv.distanceTransform(self.map, cv.DIST_L2, 3)
        else:
            raise NotImplementedError

    def bruteForce(self, ord=2) -> None:
        for i in tqdm(range(self.rows)):
            for j in range(self.cols):
                if self.map[i][j] == 0:
                    self.dist[i][j] = 0
                    continue

                point_1 = np.array([i, j])
                for ni in range(self.rows):
                    for nj in range(self.cols):
                        point_2 = np.array([ni, nj])
                        if np.all(point_1 == point_2):
                            continue

                        if self.map[ni][nj] == 1:
                            continue

                        self.dist[i][j] = min(self.dist[i][j], self.getDist(point_1, point_2, ord=ord))

    def forwardBFS(self) -> None:
        for i in tqdm(range(self.rows)):
            for j in range(self.cols):
                if self.map[i][j] == 0:
                    self.dist[i][j] = 0
                    continue

                q = queue.Queue()
                visited = np.zeros_like(self.map)

                q.put((i, j))
                visited[i][j] = 1

                step = 0
                found = False
                while not q.empty() and not found:
                    size = q.qsize()
                    for _ in range(size):
                        ci, cj = q.get()
                        if self.map[ci][cj] == 0:
                            self.dist[i][j] = step
                            found = True
                            break

                        for dir in self.dirs:
                            ni, nj = ci + dir[0], cj + dir[1]
                            if ni < 0 or ni >= self.rows or nj < 0 or nj >= self.cols or visited[ni][nj]:
                                continue

                            q.put((ni, nj))
                            visited[ni][nj] = 1

                    step += 1

    def backwardBFS(self) -> None:
        q = queue.Queue()
        visited = np.zeros_like(self.map)

        for i in range(self.rows):
            for j in range(self.cols):
                if self.map[i][j] == 0:
                    q.put((i, j))
                    visited[i][j] = 1

        step = 0
        while not q.empty():
            size = q.qsize()
            for _ in range(size):
                ci, cj = q.get()
                self.dist[ci][cj] = step

                for dir in self.dirs:
                    ni, nj = ci + dir[0], cj + dir[1]
                    if ni < 0 or ni >= self.rows or nj < 0 or nj >= self.cols or visited[ni][nj]:
                        continue

                    q.put((ni, nj))
                    visited[ni][nj] = 1

            step += 1

    def sweepDP(self) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                if self.map[i][j] == 0:
                    self.dist[i][j] = 0

        for i in tqdm(range(self.rows)):
            for j in range(self.cols):
                if i - 1 >= 0:
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i - 1][j] + 1)

                if j - 1 >= 0:
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i][j - 1] + 1)

                # if i - 1 >= 0 and j - 1 >= 0:
                #     self.dist[i][j] = min(self.dist[i][j], self.dist[i - 1][j - 1] + np.sqrt(2))

        for i in tqdm(range(self.rows - 1, -1, -1)):
            for j in range(self.cols - 1, -1, -1):
                if i + 1 < self.rows:
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i + 1][j] + 1)

                if j + 1 < self.cols:
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i][j + 1] + 1)

                # if i + 1 < self.rows and j + 1 < self.cols:
                #     self.dist[i][j] = min(self.dist[i][j], self.dist[i + 1][j + 1] + np.sqrt(2))

    def DTSampleFunction(self) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                if self.map[i][j] == 0:
                    self.dist[i][j] = 0

        # minima = np.zeros(self.map.shape)  # NOTE: zeros ?
        # bounds = np.full(self.map.shape, np.inf)

        for i in range(self.rows):
            k = 0
            for j in range(1, self.cols):
                # s = (self.dist[i][j] + j**2) - (self.dist[i][minima[i][k]])
                pass
            pass
        pass

    def BFSwithCOC(self) -> None:
        q = queue.Queue()
        visited = np.zeros_like(self.map)

        for i in range(self.rows):
            for j in range(self.cols):
                if self.map[i][j] == 0:
                    self.dist[i][j] = 0
        pass

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
        self.backwardBFS()
        # self.sweepDP()

        # bBfs_dist = self.dist
        # self.dist = dist

        # print(np.sum(np.linalg.norm(cv_dist - bBfs_dist)))

    def show(self) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(self.map)
        axs[1].imshow(self.dist)

        plt.tight_layout()
        plt.show()


def main():
    map_file = "gridmap.bmp"
    # map_file = "gridmap_64x48.bmp"

    map = cv.imread(map_file, cv.IMREAD_GRAYSCALE)
    _, map = cv.threshold(map, 127, 1, cv.THRESH_BINARY)
    # cv.imshow("map", map)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    esdf = ESDF(map)
    esdf.updateESDF()
    esdf.show()


if __name__ == "__main__":
    main()
