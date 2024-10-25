import cv2 as cv
import numpy as np

from ESDF import ESDF


class CVDistanceTransform(ESDF):
    def __init__(self, grid: np.ndarray, has_vis=True, vis_interval=0.01, frame_interval=100, frame_rate=30, output_path="") -> None:
        super().__init__(grid=grid, has_vis=has_vis, vis_interval=vis_interval, frame_interval=frame_interval, frame_rate=frame_rate, output_path=output_path)

    def updateESDF(self, ord=2) -> None:
        if ord == 1:
            self.dist = cv.distanceTransform(self.grid, cv.DIST_L1, 3)
        elif ord == 2:
            self.dist = cv.distanceTransform(self.grid, cv.DIST_L2, 3)
        else:
            raise NotImplementedError
