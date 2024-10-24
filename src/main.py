import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from methods import BackwardBFS, BackwardBFSL2, BruteForce, CVDistanceTransform, DistanceTransform, ForwardBFS, IncreBackwardBFSL2, SweepDP


def main():
    grid_file = "assets/grid.bmp"
    new_size = (80, 60)
    # new_size = (32, 24)
    # new_size = (16, 12)

    grid = cv.imread(grid_file, cv.IMREAD_GRAYSCALE)
    grid = cv.resize(grid, new_size)

    _, grid = cv.threshold(grid, 127, 1, cv.THRESH_BINARY)
    # cv.imshow("map", grid)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # esdf = CVDistanceTransform(grid=grid, output_path="./results/")
    # esdf = BruteForce(grid=grid, output_path="./results/")
    # esdf = ForwardBFS(grid=grid, output_path="./results/")
    # esdf = BackwardBFS(grid=grid, output_path="./results/")
    # esdf = SweepDP(grid=grid, output_path="./results/")
    # esdf = DistanceTransform(grid=grid, output_path="./results/")
    # esdf = BackwardBFSL2(grid=grid, output_path="./results/")
    # esdf.updateESDF()
    # esdf.show()

    ori_grid = np.ones_like(grid, dtype=np.int32)
    esdf = IncreBackwardBFSL2(grid=ori_grid, output_path="./results/")
    grid = grid.astype(np.int32)
    esdf.updateESDF(grid)

    esdf.updateDistFig(forced=True)
    plt.pause(1)

    grid = grid.copy()
    grid[22:40, 9:22] = 1
    grid[20:28, 5:13] = 0
    esdf.updateESDF(grid)

    esdf.show()


if __name__ == "__main__":
    main()
