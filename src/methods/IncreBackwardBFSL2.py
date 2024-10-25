import queue

import imageio
import matplotlib.pyplot as plt
import numpy as np

from ESDF import ESDF
from utils import Node


class IncreBackwardBFSL2(ESDF):
    def __init__(self, grid: np.ndarray, has_vis=True, vis_interval=0.01, frame_interval=100, frame_rate=30, output_path="") -> None:
        super().__init__(grid=grid, has_vis=has_vis, vis_interval=vis_interval, frame_interval=frame_interval, frame_rate=frame_rate, output_path=output_path)

        self.node_map = [[Node(np.array([x, y])) for y in range(self.cols)] for x in range(self.rows)]
        self.head = [-1 for _ in range(self.rows * self.cols)]
        self.prev = [-1 for _ in range(self.rows * self.cols)]
        self.next = [-1 for _ in range(self.rows * self.cols)]

    def coor2idx(self, pos: np.ndarray):
        return pos[0] * self.cols + pos[1]

    def idx2coor(self, idx):
        return idx // self.cols, idx % self.cols

    def deleteFromDLL(self, coc: np.ndarray, pos: np.ndarray) -> None:
        p_idx = self.coor2idx(coc)
        if p_idx == np.inf:
            return

        c_idx = self.coor2idx(pos)

        if self.prev[c_idx] != -1:
            self.next[self.prev[c_idx]] = self.next[c_idx]
        else:
            self.head[p_idx] = self.next[c_idx]

        if self.next[c_idx] != -1:
            self.prev[self.next[c_idx]] = self.prev[c_idx]

        self.prev[c_idx] = -1
        self.next[c_idx] = -1

    def insertToDLL(self, coc: np.ndarray, pos: np.ndarray) -> None:
        p_idx = self.coor2idx(coc)
        c_idx = self.coor2idx(pos)

        if self.head[p_idx] == -1:
            self.head[p_idx] = c_idx
        else:
            self.prev[self.head[p_idx]] = c_idx
            self.next[c_idx] = self.head[p_idx]
            self.head[p_idx] = c_idx

    def updateESDF(self, new_grid: np.ndarray) -> None:
        assert new_grid.shape == self.grid.shape

        grid_diff = new_grid - self.grid
        self.grid = new_grid

        if self.has_vis:
            self.axs[0].cla()
            self.axs[0].imshow(self.grid, cmap="gray")
            plt.pause(self.vis_interval)

        insert_q = queue.Queue()
        delete_q = queue.Queue()
        update_q = queue.Queue()

        for x in range(self.rows):
            for y in range(self.cols):
                if grid_diff[x][y] == -1:
                    # 0 - 1, new occupied
                    insert_q.put((x, y))
                elif grid_diff[x][y] == 1:
                    # 1 - 0, new freed
                    delete_q.put((x, y))

        while not delete_q.empty():
            x, y = delete_q.get()

            idx = self.coor2idx(self.node_map[x][y].pos)
            c_idx = self.head[idx]

            while True:
                if c_idx == -1:
                    break

                cx, cy = self.idx2coor(c_idx)
                nxt_idx = self.next[c_idx]
                self.deleteFromDLL(self.node_map[cx][cy].coc, self.node_map[cx][cy].pos)

                self.node_map[cx][cy].dis, self.node_map[cx][cy].coc = np.inf, np.array([np.inf, np.inf])
                self.dist[cx][cy] = self.node_map[cx][cy].dis
                self.updateDistFig()

                # expand neighbours
                for dir in self.dirs:
                    nx, ny = cx + dir[0], cy + dir[1]
                    if not self.checkPos(nx, ny):
                        continue

                    if self.node_map[nx][ny].coc[0] == np.inf:
                        continue

                    if self.grid[self.node_map[nx][ny].coc[0]][self.node_map[nx][ny].coc[1]] == 1:
                        continue

                    dis = self.getDist(self.node_map[nx][ny].coc, self.node_map[cx][cy].pos)
                    if dis < self.node_map[cx][cy].dis:
                        self.node_map[cx][cy].dis, self.node_map[cx][cy].coc = dis, self.node_map[nx][ny].coc
                        self.dist[cx][cy] = self.node_map[cx][cy].dis
                        self.updateDistFig()

                if not self.node_map[cx][cy].dis == np.inf:
                    self.insertToDLL(self.node_map[cx][cy].coc, self.node_map[cx][cy].pos)
                    update_q.put((cx, cy))

                c_idx = nxt_idx

        while not insert_q.empty():
            x, y = insert_q.get()
            self.deleteFromDLL(self.node_map[x][y].coc, self.node_map[x][y].pos)

            self.node_map[x][y].dis, self.node_map[x][y].coc = 0, self.node_map[x][y].pos
            self.dist[x][y] = self.node_map[x][y].dis
            self.updateDistFig()

            self.insertToDLL(self.node_map[x][y].coc, self.node_map[x][y].pos)
            update_q.put((x, y))

        while not update_q.empty():
            cx, cy = update_q.get()

            # expand neighbours
            for dir in self.dirs:
                nx, ny = cx + dir[0], cy + dir[1]
                if not self.checkPos(nx, ny):
                    continue

                dis = self.getDist(self.node_map[nx][ny].pos, self.node_map[cx][cy].coc, ord=2)
                if dis < self.node_map[nx][ny].dis:
                    self.deleteFromDLL(self.node_map[nx][ny].coc, self.node_map[nx][ny].pos)

                    self.node_map[nx][ny].dis, self.node_map[nx][ny].coc = dis, self.node_map[cx][cy].coc
                    self.dist[nx][ny] = self.node_map[nx][ny].dis
                    self.updateDistFig()

                    self.insertToDLL(self.node_map[nx][ny].coc, self.node_map[nx][ny].pos)
                    update_q.put((nx, ny))
