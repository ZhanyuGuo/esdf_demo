import numpy as np


class Node:
    def __init__(self, pos: np.ndarray, coc: np.ndarray, dis) -> None:
        self.pos = pos
        self.coc = coc
        self.dis = dis
