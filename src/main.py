import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from methods import BackwardBFS, BackwardBFSL2, BruteForce, CVDistanceTransform, DistanceTransform, ForwardBFS, IncreBackwardBFSL2, SweepDP


def main():
    grid_file = "assets/grid.bmp"
    output_path = "./results/"
    new_size = (80, 60)
    # new_size = (64, 48)
    # new_size = (32, 24)
    # new_size = (16, 12)

    grid = cv.imread(grid_file, cv.IMREAD_GRAYSCALE)
    grid = cv.resize(grid, new_size)

    _, grid = cv.threshold(grid, 127, 1, cv.THRESH_BINARY)
    # cv.imshow("map", grid)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    ## normal
    # esdf = CVDistanceTransform(grid=grid, output_path=output_path)
    # esdf = BruteForce(grid=grid, output_path=output_path)
    # esdf = ForwardBFS(grid=grid, output_path=output_path)
    # esdf = BackwardBFS(grid=grid, output_path=output_path)
    # esdf = SweepDP(grid=grid, output_path=output_path)
    # esdf = DistanceTransform(grid=grid, output_path=output_path)
    # esdf = BackwardBFSL2(grid=grid, output_path=output_path)
    
    # esdf.updateESDF()
    # esdf.show()

    ## incremental
    ori_grid = np.ones_like(grid, dtype=np.int32)
    esdf = IncreBackwardBFSL2(grid=ori_grid, output_path=output_path)
    grid = grid.astype(np.int32)
    esdf.updateESDF(grid)

    esdf.updateDistFig(forced=True)
    plt.pause(1)
    for _ in range(30):
        esdf.frame_list.append(np.array(esdf.fig.canvas.renderer.buffer_rgba()))

    grid = grid.copy()
    grid[30:44, 12:25] = 1
    grid[46:56, 6:16] = 0
    esdf.updateESDF(grid)

    esdf.show()


if __name__ == "__main__":
    main()
