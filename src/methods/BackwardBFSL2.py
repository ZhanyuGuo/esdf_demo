import queue

import numpy as np

from ESDF import ESDF
from utils import Node


class BackwardBFSL2(ESDF):
    def __init__(self, grid: np.ndarray, has_vis=True, vis_interval=0.01, acc_rate=50, frame_rate=24, output_path="") -> None:
        super().__init__(grid=grid, has_vis=has_vis, vis_interval=vis_interval, acc_rate=acc_rate, frame_rate=frame_rate, output_path=output_path)

        self.node_map = [[Node(np.array([x, y]), np.array([np.inf, np.inf]), np.inf) for y in range(self.cols)] for x in range(self.rows)]

    def updateESDF(self) -> None:
        """
        Method 1: Priority Queue
        """
        open = queue.PriorityQueue()
        closed = np.zeros_like(self.grid)

        for x in range(self.rows):
            for y in range(self.cols):
                if self.grid[x][y] == 0:
                    self.node_map[x][y].dis = 0
                    self.node_map[x][y].coc = self.node_map[x][y].pos
                    open.put((0, x, y))

        while not open.empty():
            dis, cx, cy = open.get()
            if closed[cx][cy] == 1:
                continue

            closed[cx][cy] = 1
            self.dist[cx][cy] = dis
            self.updateDistFig()

            node_c = self.node_map[cx][cy]

            # expand neighbours
            for dir in self.dirs:
                nx, ny = cx + dir[0], cy + dir[1]
                if not self.checkPos(nx, ny) or closed[nx][ny] == 1:
                    continue

                node_n = self.node_map[nx][ny]

                dis = self.getDist(node_n.pos, node_c.coc, ord=2)
                if dis < self.node_map[nx][ny].dis:
                    self.node_map[nx][ny].dis = dis
                    self.node_map[nx][ny].coc = node_c.coc
                    open.put((dis, nx, ny))

    # def updateESDF(self) -> None:
    #     """
    #     Method 2: Queue
    #     """
    #     open = queue.Queue()

    #     for x in range(self.rows):
    #         for y in range(self.cols):
    #             if self.grid[x][y] == 0:
    #                 self.node_map[x][y].dis = 0
    #                 self.node_map[x][y].coc = self.node_map[x][y].pos

    #                 self.dist[x][y] = 0
    #                 self.updateDistFig()

    #                 open.put((0, x, y))

    #     while not open.empty():
    #         dis, cx, cy = open.get()

    #         node_c = self.node_map[cx][cy]

    #         # expand neighbours
    #         for dir in self.dirs:
    #             nx, ny = cx + dir[0], cy + dir[1]
    #             if not self.checkPos(nx, ny):
    #                 continue

    #             node_n = self.node_map[nx][ny]

    #             dis = self.getDist(node_n.pos, node_c.coc, ord=2)
    #             if dis < self.node_map[nx][ny].dis:
    #                 self.node_map[nx][ny].dis = dis
    #                 self.node_map[nx][ny].coc = node_c.coc
    #                 self.dist[nx][ny] = dis
    #                 self.updateDistFig()
    #                 open.put((dis, nx, ny))
