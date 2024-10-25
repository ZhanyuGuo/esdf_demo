import numpy as np


class Node:
    def __init__(self, pos: np.ndarray, coc: np.ndarray = np.array([np.inf, np.inf]), dis=np.inf) -> None:
        self.pos = pos
        self.coc = coc
        self.dis = dis
